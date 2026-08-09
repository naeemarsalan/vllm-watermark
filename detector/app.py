# SPDX-License-Identifier: Apache-2.0
"""Watermark detector service (Phase 3).

Wraps the EXISTING, vllm-free detectors from `vllm_watermark` (imported, not
reimplemented: `vllm_watermark.kgw.detector`, `vllm_watermark.synthid.detector`,
`vllm_watermark.keys`, `vllm_watermark.synthid.core`) behind two HTTP
surfaces:

1. The TrustyAI/FMS Guardrails Orchestrator "detectors API" contract
   (`POST /api/v1/text/contents`, `GET /health`), so this service can be
   registered as a guardrails detector on the legacy FMS path (docs/facts.md
   C5) -- NOT a claim that this is the recommended future RHOAI integration
   path. The upstream NeMo 0.23.0 action and the later internal
   RHOAI-managed NeMo 0.21.0 metadata/broker path are executed in their
   recorded scopes; external gateway pass-through, supportability, and broader
   production boundaries remain open (docs/facts.md C8/C11/D5/D6/D10). Two
   scheme-forced alias routes (`/kgw/api/v1/text/contents`,
   `/synthid/api/v1/text/contents`) are also exposed -- see "Scheme
   selection" below.
2. A direct `POST /v1/watermark/detect` endpoint returning our own
   z_score/p_value/verdict-shaped response, optionally detached-JWS-signed.

Import boundary (task requirement -- service must be importable WITHOUT
vllm installed): this module imports only `vllm_watermark.keys`,
`vllm_watermark.kgw.core`, `vllm_watermark.kgw.detector`,
`vllm_watermark.synthid.core`, `vllm_watermark.synthid.detector`, and
`vllm_watermark.request_args` -- all six verified vllm-free (no
`import vllm` at module scope; confirmed both by reading each module's own
docstring and by grepping `src/vllm_watermark/` for lines starting with
`import vllm` or `from vllm` -- the only two matches anywhere in the
package are in `kgw/processor.py` and `synthid/processor.py`, never in any
module this file imports). It deliberately does NOT import
`vllm_watermark.kgw.processor` / `vllm_watermark.synthid.processor` (the
ONLY two modules in the package that import `vllm`).

TrustyAI detectors-API schema -- FETCHED, not from memory
------------------------------------------------------------------------
Source: `trustyai-explainability/guardrails-detectors` (Apache-2.0), commit
`747a4d3ef6f7d384b73f929a0162228ad56d98de` (`main`, fetched 2026-08-08 via
`gh api repos/trustyai-explainability/guardrails-detectors/...`). Field
names below are PORTED VERBATIM (pydantic model shapes) from:

  * `detectors/common/scheme.py` -- `ContentAnalysisHttpRequest`
    (`contents: List[str]`, `detector_params: Optional[Dict]`),
    `ContentAnalysisResponse` (`start: int`, `end: int`, `text: str`,
    `detection: str`, `detection_type: str`, `score: float`,
    `evidences: Optional[List[EvidenceObj]] = None`,
    `metadata: Optional[Dict[str, Any]] = {}`), `ContentsAnalysisResponse`
    (`RootModel[List[List[ContentAnalysisResponse]]]` -- one inner list per
    submitted content string, IN ORDER), `Error` (`code: int`,
    `message: str`), `EvidenceObj`/`Evidence`/`EvidenceType`.
    https://raw.githubusercontent.com/trustyai-explainability/guardrails-detectors/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/common/scheme.py
  * `detectors/common/app.py` -- `GET /health` returns the bare string
    `"ok"` (`DetectorBaseAPI.__init__`: `self.add_api_route("/health",
    health, ...)`; `async def health(): return "ok"`); the 422 validation
    exception handler returns `{"code": 422, "message": "..."}` (matches
    the `Error` model above).
    https://raw.githubusercontent.com/trustyai-explainability/guardrails-detectors/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/common/app.py
  * `detectors/huggingface/app.py` -- route registration:
    `@app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)`,
    handler runs the detector `via run_in_threadpool` (CPU-bound work off
    the event loop -- reused below for our own CPU-bound torch scoring) and
    returns `ContentsAnalysisResponse(root=result)`.
    https://raw.githubusercontent.com/trustyai-explainability/guardrails-detectors/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/huggingface/app.py
  * `detectors/huggingface/detector.py` -- the "no detection" convention:
    `Detector.run()` appends a (possibly EMPTY) list per content string
    (`contents_analyses.append(analyses)`); a content string with no
    qualifying detection contributes `[]`, not an omitted entry or a null.
    Also: `detector_params` is read DEFENSIVELY -- an invalid/unrecognized
    value falls back to a default with a logged warning, not a 422 (see
    `_resolve_params`). We mirror this convention for `detector_params.scheme`
    / `detector_params.key_id` below (the `/v1/watermark/detect` direct
    endpoint, being a stricter typed API, is intentionally NOT this loose --
    it 422s on an invalid `scheme`).
    https://raw.githubusercontent.com/trustyai-explainability/guardrails-detectors/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/huggingface/detector.py
  * `detectors/built_in/regex_detectors.py` -- confirms the "whole-content
    span" convention for content-level (as opposed to token/span-level)
    detectors: `start=0, end=len(text)`-shaped `ContentAnalysisResponse`
    rows are a normal, existing pattern for this API, not a new invention.

Scheme selection
------------------------------------
`detector_params.scheme` ("kgw"|"synthid"), if the orchestrator forwards
`detector_params` -- else env `WATERMARK_DETECTOR_SCHEME` (default "kgw").
`detector_params.key_id` works the same way (else the configured default
watermark key). Because some orchestrator/guardrails configurations
register ONE detector endpoint per behavior and cannot forward arbitrary
`detector_params` through to us, `/kgw/api/v1/text/contents` and
`/synthid/api/v1/text/contents` alias routes are ALSO exposed -- identical
handler, scheme forced by the route itself (an explicit
`detector_params.scheme` on an alias route is ignored in favor of the
route's forced scheme; `detector_params.key_id` is still honored). Cheap
insurance: the orchestrator can register two detector entries (one per
alias route) and get correct per-scheme behavior even with zero
`detector_params` forwarding support.

Calibrated `score` mapping
------------------------------------------------------------
Every detector in this package (`kgw.detector`, `synthid.detector`) already
computes an exact one-sided upper-tail normal p-value
(`0.5 * erfc(z / sqrt(2))`) in `[0, 1]` from its z-score (see those modules'
docstrings). We report

    score = clamp(1 - p_value, 0.0, 1.0)

as the TrustyAI-contract `score: float` field (documented `example=0.8`)
and as the `/v1/watermark/detect` `score` field. This is a MONOTONE mapping
of z (p_value strictly decreases as z increases, for z > 0, under both
detectors' shared exact-normal-tail formula) that lands the already-bounded
p_value into "higher = more confidently watermarked" orientation matching
every other detector in the TrustyAI ecosystem (`score` near 1.0 = strong
detection). The `clamp` is defensive only (guards float edge cases such as
`p_value` rounding fractionally below 0.0 or above 1.0); `p_value` is
mathematically already in `[0, 1]` by construction. This is NOT a
calibrated probability of "this text was truly watermarked" in a Bayesian
sense (no prior over watermarked-vs-not is assumed anywhere in this
service) -- it is a monotone rescaling of the frequentist p-value, and is
labeled as such everywhere it appears (this docstring, the README, and the
`metadata`/response fields that also expose the raw `z_score`/`p_value` so
a caller who wants the untransformed statistic always has it).

Zero retention (Code of Practice measure -- see docs/implementation.md
Phase 3, docs/facts.md A9)
------------------------------------------------------------------------
Submitted text is NEVER logged, NEVER persisted, and NEVER included in a
raised exception's message. The only content-derived value that is ever
logged is `sha256(content.encode("utf-8")).hexdigest()[:16]` (see
`_content_digest`), alongside the resolved scheme/key_id/verdict/latency --
never the content itself, never its token ids (token ids are trivially
detokenizable back to text, so they get the same treatment as raw text: not
logged). This service adds NO request-body-logging middleware, and
`detector/tests/test_service.py` asserts both (a) `app.user_middleware` is
empty (nothing could be logging bodies) and (b) a real detection request's
raw text string never appears in `caplog`'s captured records. Detector
ACCESS CONTROL (who may call this service at all) is deployment policy, not
something this service enforces itself -- see `detector/README.md` "Threat
notes".

JWS signing of `/v1/watermark/detect` responses
--------------------------------------------------
If `SIGNING_KEY_PATH` (env) points to a PEM-encoded RSA or Ed25519 PRIVATE
key, every `/v1/watermark/detect` response is accompanied by a DETACHED JWS
(RFC 7797, `b64: false` unencoded-payload mode -- verified against the
locally installed PyJWT 2.10.1 by direct execution: RSA and Ed25519
sign/verify round trips, a tamper-detection negative case, and PEM
load/type-detection were all exercised at a Python prompt before this was
written, then pinned as `TestSigning` in `detector/tests/test_service.py`,
which is the executed, re-runnable record of this claim -- not EXPERIMENTS.md,
with the aggregate execution evidence recorded in `EXPERIMENTS.md`) over
the canonical JSON encoding of the response payload EXCLUDING the
`signature`/`signing` fields themselves (`json.dumps(payload, sort_keys=True,
separators=(",", ":"), allow_nan=False).encode("utf-8")` -- see
`_canonical_json_bytes`). The algorithm (`RS256` for an RSA key, `EdDSA`
for an Ed25519 key) is auto-detected from the loaded PEM key's Python type
at startup, never configured separately (so it can never drift from what
the key actually is). If `SIGNING_KEY_PATH` is unset, the service starts
and serves completely normally -- NEVER fail closed on a missing signing
key in dev -- with every response's `"signature": null` and
`"signing": "disabled"`. If `SIGNING_KEY_PATH` IS set but unreadable /
unparsable / an unsupported key type, startup DOES fail loudly (a
misconfigured, as opposed to absent, signing key is a deployment bug worth
surfacing immediately rather than silently downgrading to unsigned).

To verify a response's signature: take the exact JSON response body,
remove the `signature` and `signing` keys, canonicalize with
`json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)`,
then (PyJWT) `jwt.api_jws.decode_complete(token, key=<public key>,
algorithms=[<alg from the JWS header>], detached_payload=<canonical
bytes>)`. See `detector/README.md` for a runnable example.

Config via env (all optional except where noted; nothing here is ever
logged with its value if it is key material)
------------------------------------------------------------------------
    WATERMARK_KEYS / WATERMARK_KEY (+ WATERMARK_KEY_ID)
        REUSED, unchanged, from `vllm_watermark.keys.load_keys()` -- the
        exact same key material the generation-side processors load. If
        NEITHER is set, the service still starts (GET /health stays 200)
        but GET /ready reports not-ready and every detection request 503s
        with "no watermark keys configured" -- fail-loud-per-request, not
        fail-to-start, so a misconfigured detector doesn't take down a
        liveness probe (mirrors the generation-side processors' own
        graceful-degradation philosophy -- see kgw/processor.py __init__
        docstring).
    WATERMARK_DETECTOR_SCHEME   "kgw"|"synthid", default "kgw" -- see
        "Scheme selection" above.
    MODEL_TOKENIZER   HF tokenizer name or local path, default
        "Qwen/Qwen2.5-0.5B-Instruct" -- pre-loaded once at startup (lifespan),
        not per-request.
    WATERMARK_VOCAB_SIZE   int in [1, 2**20] -- MUST equal the exact vocab_size used
        at GENERATION time (`vllm_config.model_config.get_vocab_size()`,
        NOT `len(tokenizer)` -- see kgw/core.py module docstring "CRITICAL
        DEVIATION"/"vocab_size is REQUIRED" for why a mismatch here
        silently produces near-zero scores with no error raised, for BOTH
        schemes). If unset, this service falls back to
        `len(tokenizer)` at startup with a loud WARNING log, subject to the
        same [1, 2**20] and effective-KGW-greenlist safety checks (a failed
        check is a startup crash) -- this fallback is a convenience for the
        `/v1/watermark/detect` smoke-test / local-dev path, not something
        to rely on on a real deployment whose model pads its embedding
        matrix past the tokenizer's own vocab length.
    WATERMARK_Z_THRESHOLD   float in [0, 100], default 4.0 -- shared by both schemes'
        `z_threshold` (both `kgw.detector.DEFAULT_Z_THRESHOLD` and
        `synthid.detector.DEFAULT_Z_THRESHOLD` already independently
        default to 4.0 -- see those modules; one shared env knob is
        simpler than two that would almost always be set identically).
    WATERMARK_KGW_IGNORE_REPEATED_NGRAMS   "on"/"off", default "on" --
        matches the configuration actually used to produce the measured
        Phase 1 detection statistics in EXPERIMENTS.md ("Scored locally
        (CPU detector, ignore_repeated_ngrams=True, z>=4.0)").
    WATERMARK_SYNTHID_SCORER   "mean"|"weighted_mean", default
        "weighted_mean" -- matches docs/implementation.md Phase 2's
        explicit guidance ("start with the untrained weighted-mean
        scorer").
    VLLM_WATERMARK_GAMMA (0 < gamma < 1), VLLM_WATERMARK_DELTA (0 <= delta <= 100)
        REUSED, same env var NAMES and DEFAULTS ("0.25"/"2.0") as
        `vllm_watermark.kgw.processor` -- a deliberate design choice so an
        operator sets KGW params ONCE (in a shared env/ConfigMap) for both
        the generation-side plugin and this detector, instead of two
        independently-named knobs that could silently drift apart (`delta`
        itself is carried for `KGWConfig` parity only -- the KGW detector
        math never reads `.delta`).
    VLLM_WATERMARK_SYNTHID_NGRAM_LEN (1..1024),
    VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SIZE (1..2**24),
    VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SEED (-(2**63)..2**64-1),
    VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE (0..2**16),
    VLLM_WATERMARK_SYNTHID_KEY_DEPTH (1..256)
        REUSED, same env var NAMES and DEFAULTS as
        `vllm_watermark.synthid.processor` (same rationale as gamma/delta
        above -- SynthID g-values silently disagree between generation and
        detection if ANY of these five differ, per synthid/core.py's module
        docstring). `VLLM_WATERMARK_SYNTHID_KEY_DEPTH` defaults to
        `vllm_watermark.synthid.core.DEFAULT_SYNTHID_DEPTH` (30) and subkeys
        are derived with `vllm_watermark.synthid.core.SYNTHID_KEY_LABEL` --
        the EXACT constants the generation side uses.
    SIGNING_KEY_PATH, SIGNING_KEY_ID   see "JWS signing" above.
        SIGNING_KEY_ID (optional) becomes the JWS header's `kid` claim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from starlette.concurrency import run_in_threadpool
from transformers import AutoTokenizer

import vllm_watermark
from vllm_watermark.keys import WatermarkKey, load_key, load_keys
from vllm_watermark.kgw.core import KGWConfig
from vllm_watermark.kgw.detector import (
    DEFAULT_Z_THRESHOLD as KGW_DEFAULT_Z_THRESHOLD,
    score_token_ids as kgw_score_token_ids,
)
from vllm_watermark.request_args import VALID_SCHEMES
from vllm_watermark.synthid.core import DEFAULT_SYNTHID_DEPTH, SYNTHID_KEY_LABEL, SynthIDConfig
from vllm_watermark.synthid.detector import (
    DEFAULT_Z_THRESHOLD as SYNTHID_DEFAULT_Z_THRESHOLD,
    score_token_ids_mean,
    score_token_ids_weighted_mean,
)

logger = logging.getLogger("vllm_watermark.detector")

DETECTOR_VERSION = f"vllm-watermark-detector/{vllm_watermark.__version__}"

_DEFAULT_MODEL_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
_DEFAULT_DETECTOR_SCHEME = "kgw"
_DEFAULT_Z_THRESHOLD_ENV = "4.0"
_DEFAULT_KGW_GAMMA_ENV = "0.25"
_DEFAULT_KGW_DELTA_ENV = "2.0"
_DEFAULT_KGW_IGNORE_REPEATED_NGRAMS_ENV = "on"
_DEFAULT_SYNTHID_SCORER = "weighted_mean"
_VALID_SYNTHID_SCORERS = ("mean", "weighted_mean")
_DEFAULT_SYNTHID_NGRAM_LEN_ENV = "5"
_DEFAULT_SYNTHID_SAMPLING_TABLE_SIZE_ENV = str(1 << 16)
_DEFAULT_SYNTHID_SAMPLING_TABLE_SEED_ENV = "0"
_DEFAULT_SYNTHID_CONTEXT_HISTORY_SIZE_ENV = "1024"
_DEFAULT_SYNTHID_KEY_DEPTH_ENV = str(DEFAULT_SYNTHID_DEPTH)
# These are service-side safety caps, not algorithmic requirements.  They
# bound the work an operator can request through environment configuration:
# KGW allocates an int64 permutation proportional to vocab_size, SynthID
# retains ngram/context-history state proportional to these settings, and
# SynthID's g-value path does work proportional to key depth.  The caps leave
# ample headroom over the recorded deployment (vocab 151,936; ngram 5;
# history 1,024; depth 30) while preventing accidental host-memory/CPU blowups.
_MAX_WATERMARK_VOCAB_SIZE = 1 << 20  # 1,048,576 int64 ids (~8 MiB per permutation)
_MAX_WATERMARK_Z_THRESHOLD = 100.0  # far beyond ordinary detector z-scores
_MAX_KGW_DELTA = 100.0  # larger logit biases are numerically saturated in practice
_MAX_SYNTHID_NGRAM_LEN = 1 << 10  # bounds each context tuple to 1,024 token ids
_MAX_SYNTHID_CONTEXT_HISTORY_SIZE = 1 << 16  # bounds the per-request history window
_MAX_SYNTHID_KEY_DEPTH = 1 << 8  # bounds per-token tournament layers/key derivation
_MAX_SYNTHID_SAMPLING_TABLE_SIZE = 1 << 24
_MIN_TORCH_SEED = -(1 << 63)
_MAX_TORCH_SEED = (1 << 64) - 1

# Sanity check (module import time, not a runtime env read): both schemes'
# own DEFAULT_Z_THRESHOLD constants already agree at 4.0 -- see module
# docstring "WATERMARK_Z_THRESHOLD" for why one shared knob is deliberate,
# not an oversight. If a future change to either module's constant ever
# makes them disagree, this assertion is a loud reminder to reconsider that
# design choice rather than silently going stale.
assert KGW_DEFAULT_Z_THRESHOLD == SYNTHID_DEFAULT_Z_THRESHOLD == 4.0


def _content_digest(content: str) -> str:
    """First 16 hex chars of sha256(content) -- the ONLY content-derived
    value this service ever logs. See module docstring "Zero retention"."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _content_sha256(content: str) -> str:
    """Full SHA-256 used only for API correlation, never log output."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _calibrated_score(p_value: float) -> float:
    """score = clamp(1 - p_value, 0, 1) -- see module docstring "Calibrated
    score mapping" for the full rationale."""
    return max(0.0, min(1.0, 1.0 - p_value))


def _canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON encoding used both to sign and (by a verifier, per
    module docstring "JWS signing") to re-derive the exact bytes a detached
    JWS signature was computed over. `allow_nan=False` is deliberate: NaN/
    Infinity are not valid JSON, so refusing to silently emit them here
    means a pathological score can never produce a canonical encoding whose
    round-trip through a standards-compliant JSON parser disagrees with
    what was actually signed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class InsufficientTokensError(ValueError):
    """Raised when a text has too few (post-tokenization) tokens for the
    requested scheme to score at all (KGW: < 2 tokens; SynthID: < ngram_len
    tokens, or every scoreable position was masked out as a repeated
    context). Message text is derived from the underlying detector's own
    ValueError, which never embeds the raw text (only counts) -- see module
    docstring "Zero retention"."""


# ---------------------------------------------------------------------------
# Settings (env parsing, read once at service startup -- see `lifespan`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    default_scheme: str
    model_tokenizer: str
    vocab_size_env: Optional[int]
    z_threshold: float
    kgw_gamma: float
    kgw_delta: float
    kgw_ignore_repeated_ngrams: bool
    synthid_scorer: str
    synthid_ngram_len: int
    synthid_sampling_table_size: int
    synthid_sampling_table_seed: int
    synthid_context_history_size: int
    synthid_key_depth: int
    signing_key_path: Optional[str]
    signing_key_id: Optional[str]


def _env_bool(env: Mapping[str, str], name: str, default: str) -> bool:
    v = env.get(name, default).strip().lower()
    if v in ("on", "true", "1", "yes"):
        return True
    if v in ("off", "false", "0", "no"):
        return False
    raise RuntimeError(f"{name} must be 'on'/'off' (or a boolean-like string), got {v!r}")


def _env_int(
    env: Mapping[str, str], name: str, default: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    """Read a bounded integer env setting without deferring invalid config
    until request handling. Bounds are inclusive when supplied."""
    raw = env.get(name, default)
    if not isinstance(raw, str):
        raise RuntimeError(f"{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{name} must be <= {maximum}, got {value}")
    return value


def _env_finite_float(
    env: Mapping[str, str], name: str, default: str, *, minimum: float | None = None,
    minimum_exclusive: bool = False, maximum: float | None = None, maximum_exclusive: bool = False,
) -> float:
    """Read a finite bounded float env setting.

    Rejecting NaN/Infinity at startup is essential: both can otherwise pass
    ordinary comparisons and create a detector which is ready but unusable.
    """
    raw = env.get(name, default)
    if not isinstance(raw, str):
        raise RuntimeError(f"{name} must be a finite number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be a finite number")
    if minimum is not None and (value <= minimum if minimum_exclusive else value < minimum):
        operator = ">" if minimum_exclusive else ">="
        raise RuntimeError(f"{name} must be {operator} {minimum}, got {value}")
    if maximum is not None and (value >= maximum if maximum_exclusive else value > maximum):
        operator = "<" if maximum_exclusive else "<="
        raise RuntimeError(f"{name} must be {operator} {maximum}, got {value}")
    return value


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Parse all env-driven config into a `Settings` instance. Pure
    function of `env` (defaults to `os.environ`) -- no I/O, no side
    effects, so tests can call this directly against a synthetic mapping."""
    env = env if env is not None else os.environ

    default_scheme = env.get("WATERMARK_DETECTOR_SCHEME", _DEFAULT_DETECTOR_SCHEME).strip().lower()
    if default_scheme not in VALID_SCHEMES:
        raise RuntimeError(
            f"WATERMARK_DETECTOR_SCHEME must be one of {VALID_SCHEMES!r}, got {default_scheme!r}"
        )

    synthid_scorer = env.get("WATERMARK_SYNTHID_SCORER", _DEFAULT_SYNTHID_SCORER).strip().lower()
    if synthid_scorer not in _VALID_SYNTHID_SCORERS:
        raise RuntimeError(
            f"WATERMARK_SYNTHID_SCORER must be one of {_VALID_SYNTHID_SCORERS!r}, got {synthid_scorer!r}"
        )

    vocab_size_env: Optional[int] = None
    if "WATERMARK_VOCAB_SIZE" in env:
        raw_vocab = env.get("WATERMARK_VOCAB_SIZE")
        if not isinstance(raw_vocab, str):
            raise RuntimeError("WATERMARK_VOCAB_SIZE must be an integer")
        vocab_size_env = _env_int(
            env, "WATERMARK_VOCAB_SIZE", raw_vocab, minimum=1, maximum=_MAX_WATERMARK_VOCAB_SIZE
        )

    settings = Settings(
        default_scheme=default_scheme,
        model_tokenizer=env.get("MODEL_TOKENIZER", _DEFAULT_MODEL_TOKENIZER),
        vocab_size_env=vocab_size_env,
        z_threshold=_env_finite_float(
            env, "WATERMARK_Z_THRESHOLD", _DEFAULT_Z_THRESHOLD_ENV,
            minimum=0.0, maximum=_MAX_WATERMARK_Z_THRESHOLD,
        ),
        kgw_gamma=_env_finite_float(
            env, "VLLM_WATERMARK_GAMMA", _DEFAULT_KGW_GAMMA_ENV,
            minimum=0.0, minimum_exclusive=True, maximum=1.0, maximum_exclusive=True,
        ),
        kgw_delta=_env_finite_float(
            env, "VLLM_WATERMARK_DELTA", _DEFAULT_KGW_DELTA_ENV,
            minimum=0.0, maximum=_MAX_KGW_DELTA,
        ),
        kgw_ignore_repeated_ngrams=_env_bool(
            env, "WATERMARK_KGW_IGNORE_REPEATED_NGRAMS", _DEFAULT_KGW_IGNORE_REPEATED_NGRAMS_ENV
        ),
        synthid_scorer=synthid_scorer,
        synthid_ngram_len=_env_int(
            env, "VLLM_WATERMARK_SYNTHID_NGRAM_LEN", _DEFAULT_SYNTHID_NGRAM_LEN_ENV,
            minimum=1, maximum=_MAX_SYNTHID_NGRAM_LEN,
        ),
        synthid_sampling_table_size=_env_int(
            env, "VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SIZE",
            _DEFAULT_SYNTHID_SAMPLING_TABLE_SIZE_ENV, minimum=1, maximum=_MAX_SYNTHID_SAMPLING_TABLE_SIZE,
        ),
        synthid_sampling_table_seed=_env_int(
            env, "VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SEED",
            _DEFAULT_SYNTHID_SAMPLING_TABLE_SEED_ENV, minimum=_MIN_TORCH_SEED, maximum=_MAX_TORCH_SEED,
        ),
        synthid_context_history_size=_env_int(
            env, "VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE",
            _DEFAULT_SYNTHID_CONTEXT_HISTORY_SIZE_ENV, minimum=0,
            maximum=_MAX_SYNTHID_CONTEXT_HISTORY_SIZE,
        ),
        synthid_key_depth=_env_int(
            env, "VLLM_WATERMARK_SYNTHID_KEY_DEPTH", _DEFAULT_SYNTHID_KEY_DEPTH_ENV,
            minimum=1, maximum=_MAX_SYNTHID_KEY_DEPTH,
        ),
        signing_key_path=env.get("SIGNING_KEY_PATH") or None,
        signing_key_id=env.get("SIGNING_KEY_ID") or None,
    )
    if settings.vocab_size_env is not None:
        _validate_kgw_greenlist_size(settings.vocab_size_env, settings.kgw_gamma)
    return settings


def _validate_kgw_greenlist_size(vocab_size: int, gamma: float) -> None:
    """Reject a valid-looking pair that would construct an empty greenlist."""
    greenlist_size = int(vocab_size * gamma)
    if greenlist_size < 1:
        raise RuntimeError(
            "effective KGW greenlist_size must be >= 1; increase WATERMARK_VOCAB_SIZE "
            "or VLLM_WATERMARK_GAMMA"
        )


def _validate_tokenizer_vocab_size(vocab_size: int, gamma: float) -> int:
    """Validate the tokenizer-derived fallback using the same startup rules."""
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise RuntimeError("tokenizer vocabulary size must be an integer")
    if not 1 <= vocab_size <= _MAX_WATERMARK_VOCAB_SIZE:
        raise RuntimeError(
            "tokenizer vocabulary size must be in [1, "
            f"{_MAX_WATERMARK_VOCAB_SIZE}]"
        )
    _validate_kgw_greenlist_size(vocab_size, gamma)
    return vocab_size


def _load_signing_key(path: Optional[str]):
    """Returns (key_object, alg) or (None, None) if `path` is falsy. Never
    logs key material (not even the path's contents -- only the fact that a
    path was/wasn't configured is ever logged by the caller). Raises loudly
    (see module docstring "JWS signing") if `path` IS set but unreadable or
    an unsupported key type -- a configured-but-broken signing key is a
    deployment bug, not a "run unsigned" situation."""
    if not path:
        return None, None
    pem_bytes = Path(path).read_bytes()
    try:
        key_obj = serialization.load_pem_private_key(pem_bytes, password=None)
    finally:
        pem_bytes = b""  # best-effort: don't keep the raw PEM bytes around longer than needed
    if isinstance(key_obj, rsa.RSAPrivateKey):
        return key_obj, "RS256"
    if isinstance(key_obj, ed25519.Ed25519PrivateKey):
        return key_obj, "EdDSA"
    raise RuntimeError(
        f"SIGNING_KEY_PATH={path!r} is a PEM private key of unsupported type "
        f"{type(key_obj).__name__}; only RSA and Ed25519 private keys are supported"
    )


def _sign_payload(
    payload: Dict[str, Any], signing_key, signing_alg: Optional[str], kid: Optional[str]
) -> "tuple[Optional[str], str]":
    """Returns (signature, signing_marker). See module docstring "JWS
    signing" -- detached (RFC 7797, b64:false) JWS over
    `_canonical_json_bytes(payload)`. `payload` must NOT already contain
    "signature"/"signing" keys (the caller adds those AFTER calling this,
    from this function's own return value)."""
    if signing_key is None:
        return None, "disabled"
    canonical = _canonical_json_bytes(payload)
    headers: Dict[str, Any] = {"crit": ["b64"], "typ": "JOSE"}
    if kid:
        headers["kid"] = kid
    token = jwt.api_jws.encode(
        canonical, signing_key, algorithm=signing_alg, headers=headers, is_payload_detached=True
    )
    return token, "enabled"


# ---------------------------------------------------------------------------
# TrustyAI detectors-API pydantic schema -- PORTED VERBATIM from
# detectors/common/scheme.py at the fetched commit (see module docstring
# "TrustyAI detectors-API schema" for the exact citation). Apache-2.0,
# trustyai-explainability/guardrails-detectors; attributed per AGENTS.md
# licensing rules. Field names/types/defaults are intentionally identical
# to the fetched source, not paraphrased.
# ---------------------------------------------------------------------------


class EvidenceType(str, Enum):
    url = "url"
    title = "title"


class Evidence(BaseModel):
    source: str = Field(
        title="Source",
        description="Source of the evidence, it can be url of the evidence etc",
    )


class EvidenceObj(BaseModel):
    type: EvidenceType = Field(
        title="EvidenceType",
        description="Type field signifying the type of evidence provided. Example url, title etc",
    )
    evidence: Evidence = Field(
        description=(
            "Evidence object, currently only containing source, but in future can "
            "contain other optional arguments like id, etc"
        ),
    )


class ContentAnalysisHttpRequest(BaseModel):
    contents: List[str] = Field(
        min_length=1,
        title="Contents",
        description=(
            "Field allowing users to provide list of texts for analysis. Note, "
            "results of this endpoint will contain analysis / detection of each "
            "of the provided text in the order they are present in the contents "
            "object."
        ),
    )
    detector_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional detector parameters, used on a per-detector basis"
    )


class ContentAnalysisResponse(BaseModel):
    # `json_schema_extra={"example": ...}` (not pydantic v1's bare
    # `example=` kwarg, which the fetched source uses but pydantic v2
    # deprecates -- see PydanticDeprecatedSince20) reproduces the identical
    # OpenAPI example values without a deprecation warning under the
    # pydantic 2.12.5 this service actually runs (see requirements.txt).
    start: int = Field(json_schema_extra={"example": 0})
    end: int = Field(json_schema_extra={"example": 26})
    text: str = Field(json_schema_extra={"example": "abc@def.com"})
    detection: str = Field(default="detection", json_schema_extra={"example": "kgw-watermark"})
    detection_type: str = Field(json_schema_extra={"example": "watermark"})
    score: float = Field(json_schema_extra={"example": 0.8})
    evidences: Optional[List[EvidenceObj]] = Field(
        default=None,
        description="Optional field providing evidences for the provided detection",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata from evaluation"
    )


class ContentsAnalysisResponse(RootModel):
    root: List[List[ContentAnalysisResponse]] = Field(
        title="Response Text Content Analysis Unary Handler Api V1 Text Content Post"
    )


class Error(BaseModel):
    code: int
    message: str


# ---------------------------------------------------------------------------
# Detection core (scheme-agnostic; shared by both HTTP surfaces)
# ---------------------------------------------------------------------------


def _kgw_config(key: WatermarkKey, settings: Settings, vocab_size: int) -> KGWConfig:
    return KGWConfig(
        vocab_size=vocab_size, hash_key=key.hash_key, gamma=settings.kgw_gamma, delta=settings.kgw_delta
    )


def _synthid_config(key: WatermarkKey, settings: Settings, vocab_size: int) -> SynthIDConfig:
    # MUST match the generation-side (depth, label) exactly -- see module
    # docstring "VLLM_WATERMARK_SYNTHID_KEY_DEPTH" and
    # vllm_watermark.synthid.core.SYNTHID_KEY_LABEL's own docstring.
    subkeys = key.derive_subkeys(settings.synthid_key_depth, SYNTHID_KEY_LABEL)
    return SynthIDConfig(
        vocab_size=vocab_size,
        keys=subkeys,
        ngram_len=settings.synthid_ngram_len,
        sampling_table_size=settings.synthid_sampling_table_size,
        sampling_table_seed=settings.synthid_sampling_table_seed,
        context_history_size=settings.synthid_context_history_size,
    )


def score_token_ids(
    token_ids: List[int], scheme: str, key: WatermarkKey, settings: Settings, vocab_size: int
) -> Dict[str, Any]:
    """Score `token_ids` with the requested scheme's EXISTING detector
    (imported, not reimplemented -- see module docstring "Import
    boundary"). Returns a normalized dict:
        num_tokens_scored, z_score, p_value, verdict (bool), scheme_details
    (scheme_details carries the scheme-specific extras -- num_green/gamma
    for kgw; mean_g/score/depth/scorer for synthid.)

    Raises:
        InsufficientTokensError: too few tokens to score at all (see that
            class's docstring).
        ValueError: unsupported `scheme`.
    """
    if scheme == "kgw":
        cfg = _kgw_config(key, settings, vocab_size)
        try:
            result = kgw_score_token_ids(
                token_ids,
                cfg,
                ignore_repeated_ngrams=settings.kgw_ignore_repeated_ngrams,
                z_threshold=settings.z_threshold,
            )
        except ValueError as exc:
            raise InsufficientTokensError(f"kgw: {exc}") from exc
        return {
            "num_tokens_scored": result.num_tokens_scored,
            "z_score": result.z_score,
            "p_value": result.p_value,
            "verdict": result.prediction,
            "scheme_details": {"num_green": result.num_green, "gamma": cfg.gamma},
        }

    if scheme == "synthid":
        cfg = _synthid_config(key, settings, vocab_size)
        scorer_fn = (
            score_token_ids_weighted_mean if settings.synthid_scorer == "weighted_mean" else score_token_ids_mean
        )
        try:
            result = scorer_fn(token_ids, cfg, z_threshold=settings.z_threshold)
        except ValueError as exc:
            raise InsufficientTokensError(f"synthid: {exc}") from exc
        return {
            "num_tokens_scored": result.num_scored,
            "z_score": result.z_score,
            "p_value": result.p_value,
            "verdict": result.prediction,
            "scheme_details": {
                "mean_g": result.mean_g,
                "score": result.score,
                "depth": result.depth,
                "scorer": settings.synthid_scorer,
            },
        }

    raise ValueError(f"unsupported scheme {scheme!r}; must be one of {VALID_SCHEMES!r}")


def _resolve_scheme_from_params(
    detector_params: Optional[Dict[str, Any]], default_scheme: str, forced_scheme: Optional[str]
) -> str:
    """See module docstring "Scheme selection". `forced_scheme` (set by the
    `/kgw/.../contents` and `/synthid/.../contents` alias routes) always
    wins. Otherwise reads `detector_params["scheme"]` DEFENSIVELY (TrustyAI
    convention -- see module docstring citation of
    `detectors/huggingface/detector.py`'s `_resolve_params`): an invalid
    value logs a warning and falls back to `default_scheme` rather than
    422ing the whole request."""
    if forced_scheme is not None:
        return forced_scheme
    detector_params = detector_params or {}
    raw = detector_params.get("scheme")
    if raw is None:
        return default_scheme
    if isinstance(raw, str) and raw.strip().lower() in VALID_SCHEMES:
        return raw.strip().lower()
    logger.warning("detector_params.scheme=%r is invalid; falling back to default %r", raw, default_scheme)
    return default_scheme


def _resolve_key_id_from_params(detector_params: Optional[Dict[str, Any]]) -> Optional[str]:
    """See `_resolve_scheme_from_params` docstring -- same defensive
    convention for `detector_params["key_id"]`."""
    detector_params = detector_params or {}
    raw = detector_params.get("key_id")
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    logger.warning("detector_params.key_id=%r is invalid (must be a non-empty string); ignoring", raw)
    return None


def _resolve_key(
    key_id: Optional[str], keys: Dict[str, WatermarkKey], default_key: Optional[WatermarkKey]
) -> WatermarkKey:
    if key_id is None:
        if default_key is None:
            raise KeyError("no default watermark key is configured")
        return default_key
    if key_id not in keys:
        raise KeyError(f"key_id={key_id!r} not found among configured keys")
    return keys[key_id]


# ---------------------------------------------------------------------------
# /api/v1/text/contents (+ scheme-forced aliases)
# ---------------------------------------------------------------------------


def _analyze_contents_sync(
    contents: List[str],
    detector_params: Optional[Dict[str, Any]],
    forced_scheme: Optional[str],
    state: Any,
) -> List[List[ContentAnalysisResponse]]:
    """Synchronous (CPU-bound: tokenization + torch scoring) core, run via
    `run_in_threadpool` by the async route handlers below -- mirrors the
    fetched `detectors/huggingface/app.py` convention of keeping the event
    loop free during detector inference (see module docstring citation).
    """
    settings: Settings = state.settings
    keys: Dict[str, WatermarkKey] = state.keys
    if not keys:
        raise HTTPException(status_code=503, detail="no watermark keys configured")

    scheme = _resolve_scheme_from_params(detector_params, settings.default_scheme, forced_scheme)
    key_id = _resolve_key_id_from_params(detector_params)
    try:
        key = _resolve_key(key_id, keys, state.default_key)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results: List[List[ContentAnalysisResponse]] = []
    for content in contents:
        t0 = time.monotonic()
        digest = _content_digest(content)
        token_ids = state.tokenizer.encode(content, add_special_tokens=False)

        try:
            scored = score_token_ids(token_ids, scheme, key, settings, state.vocab_size)
        except InsufficientTokensError:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "contents_detect content_sha256_16=%s scheme=%s key_id=%s "
                "verdict=skip(insufficient_tokens) latency_ms=%.2f",
                digest,
                scheme,
                key.key_id,
                latency_ms,
            )
            results.append([])
            continue

        latency_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "contents_detect content_sha256_16=%s scheme=%s key_id=%s verdict=%s z=%.3f latency_ms=%.2f",
            digest,
            scheme,
            key.key_id,
            scored["verdict"],
            scored["z_score"],
            latency_ms,
        )

        if not scored["verdict"]:
            # Below-threshold -> empty list for this content. See module
            # docstring "TrustyAI detectors-API schema" citation of
            # detectors/huggingface/detector.py's run() -- this is the
            # fetched-and-verified "no detection" convention, not a guess.
            results.append([])
            continue

        results.append(
            [
                ContentAnalysisResponse(
                    start=0,
                    end=len(content),
                    text=content,
                    detection=f"{scheme}-watermark",
                    detection_type="watermark",
                    score=_calibrated_score(scored["p_value"]),
                    evidences=None,
                    metadata={
                        "z_score": scored["z_score"],
                        "p_value": scored["p_value"],
                        "key_id": key.key_id,
                        "scheme": scheme,
                        "num_tokens_scored": scored["num_tokens_scored"],
                        "detector_version": DETECTOR_VERSION,
                        **scored["scheme_details"],
                    },
                )
            ]
        )
    return results


# ---------------------------------------------------------------------------
# /v1/watermark/detect (direct endpoint)
# ---------------------------------------------------------------------------


class DetectRequest(BaseModel):
    """Exactly one of `text`/`texts` must be provided (enforced below --
    FastAPI turns the resulting pydantic ValidationError into an HTTP 422,
    satisfying the "malformed body -> 422" requirement). Unlike
    `detector_params` on the TrustyAI-contract routes (defensive, warn +
    fall back), THIS is a strict typed API: an invalid `scheme` value is a
    422, not a silent fallback -- callers of a direct, purpose-built
    endpoint get precise errors instead of guessing why their explicit
    choice was ignored."""

    text: Optional[str] = None
    texts: Optional[List[str]] = None
    key_id: Optional[str] = None
    scheme: Optional[str] = None
    validation_id: Optional[str] = None
    response_id: Optional[str] = None

    @field_validator("scheme")
    @classmethod
    def _validate_scheme(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        normalized = v.strip().lower()
        if normalized not in VALID_SCHEMES:
            raise ValueError(f"scheme must be one of {VALID_SCHEMES!r}, got {v!r}")
        return normalized

    @field_validator("validation_id")
    @classmethod
    def _validate_validation_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", v
        ):
            raise ValueError("validation_id must be a canonical lowercase UUID string")
        try:
            parsed = UUID(v)
        except ValueError as exc:
            raise ValueError("validation_id must be a canonical lowercase UUID string") from exc
        if str(parsed) != v:
            raise ValueError("validation_id must be a canonical lowercase UUID string")
        return v

    @field_validator("response_id")
    @classmethod
    def _validate_response_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", v):
            raise ValueError("response_id must be a bounded safe identifier")
        return v

    @model_validator(mode="after")
    def _check_text_xor_texts(self) -> "DetectRequest":
        if (self.text is None) == (self.texts is None):
            raise ValueError("exactly one of 'text' or 'texts' must be provided")
        if self.texts is not None and len(self.texts) == 0:
            raise ValueError("'texts' must be a non-empty list")
        if self.texts is not None and (self.validation_id is not None or self.response_id is not None):
            raise ValueError("validation_id and response_id are supported only with a single 'text'")
        return self


def _build_detect_result(
    text: str,
    scheme: str,
    key: WatermarkKey,
    settings: Settings,
    state: Any,
    validation_id: Optional[str] = None,
    response_id: Optional[str] = None,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    digest = _content_digest(text)
    content_sha256 = _content_sha256(text)
    t0 = time.monotonic()
    token_ids = state.tokenizer.encode(text, add_special_tokens=False)
    try:
        scored = score_token_ids(token_ids, scheme, key, settings, state.vocab_size)
    except InsufficientTokensError as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "watermark_detect validation_id=%s content_sha256_16=%s scheme=%s key_id=%s "
            "verdict=error(insufficient_tokens) latency_ms=%.2f",
            validation_id or "-",
            digest,
            scheme,
            key.key_id,
            latency_ms,
        )
        where = f" (texts[{index}])" if index is not None else ""
        raise HTTPException(
            status_code=422, detail=f"text too short to score for scheme {scheme!r}{where}: {exc}"
        ) from exc

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "watermark_detect validation_id=%s content_sha256_16=%s scheme=%s key_id=%s "
        "verdict=%s z=%.3f latency_ms=%.2f",
        validation_id or "-",
        digest,
        scheme,
        key.key_id,
        scored["verdict"],
        scored["z_score"],
        latency_ms,
    )

    return {
        "validation_id": validation_id,
        "response_id": response_id,
        "content_sha256": content_sha256,
        "scheme": scheme,
        "key_id": key.key_id,
        "verdict": scored["verdict"],
        "z_score": scored["z_score"],
        "p_value": scored["p_value"],
        "score": _calibrated_score(scored["p_value"]),
        "num_tokens_scored": scored["num_tokens_scored"],
        "detector_version": DETECTOR_VERSION,
        "model_tokenizer": settings.model_tokenizer,
        "scheme_details": scored["scheme_details"],
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings

    logger.info("loading tokenizer %s", settings.model_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_tokenizer)
    app.state.tokenizer = tokenizer

    if settings.vocab_size_env is not None:
        app.state.vocab_size = settings.vocab_size_env
    else:
        app.state.vocab_size = _validate_tokenizer_vocab_size(len(tokenizer), settings.kgw_gamma)
        logger.warning(
            "WATERMARK_VOCAB_SIZE not set; falling back to len(tokenizer)=%d. This "
            "MUST match the exact vocab_size used at generation time or scores will "
            "be silently near-zero -- see app.py module docstring 'WATERMARK_VOCAB_SIZE'.",
            app.state.vocab_size,
        )

    try:
        app.state.keys = load_keys()
    except RuntimeError as exc:
        logger.warning("no watermark keys configured at startup (%s); /ready will 503 until fixed", exc)
        app.state.keys = {}
    try:
        app.state.default_key = load_key(key_id=None) if app.state.keys else None
    except (RuntimeError, KeyError):
        app.state.default_key = None

    signing_key, signing_alg = _load_signing_key(settings.signing_key_path)
    app.state.signing_key = signing_key
    app.state.signing_alg = signing_alg
    logger.info("signing: %s", "enabled" if signing_key is not None else "disabled")

    yield
    # Nothing to release: no file handles or GPU memory held past startup.


def create_app() -> FastAPI:
    """Factory (rather than a bare module-level `app = FastAPI(...)`) so
    tests can build a fresh app -- with its OWN `lifespan`-time env read --
    per test scenario via `monkeypatch.setenv(...)` followed by
    `with TestClient(create_app()) as client: ...`. `uvicorn app:app` /
    `uvicorn detector.app:app` still works via the module-level `app`
    instance below."""
    app = FastAPI(
        title="vllm-watermark detector",
        version=DETECTOR_VERSION,
        lifespan=_lifespan,
    )

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        state = request.app.state
        tokenizer_ready = getattr(state, "tokenizer", None) is not None
        keys = getattr(state, "keys", None) or {}
        default_key = getattr(state, "default_key", None)
        if tokenizer_ready and default_key is not None:
            return {"status": "ready", "tokenizer_loaded": True, "key_ids": sorted(keys)}
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "tokenizer_loaded": tokenizer_ready,
                "keys_configured": bool(keys),
                "default_key_configured": default_key is not None,
            },
        )

    @app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)
    async def contents_generic(req: ContentAnalysisHttpRequest, request: Request) -> ContentsAnalysisResponse:
        results = await run_in_threadpool(
            _analyze_contents_sync, req.contents, req.detector_params, None, request.app.state
        )
        return ContentsAnalysisResponse(root=results)

    @app.post("/kgw/api/v1/text/contents", response_model=ContentsAnalysisResponse)
    async def contents_kgw(req: ContentAnalysisHttpRequest, request: Request) -> ContentsAnalysisResponse:
        results = await run_in_threadpool(
            _analyze_contents_sync, req.contents, req.detector_params, "kgw", request.app.state
        )
        return ContentsAnalysisResponse(root=results)

    @app.post("/synthid/api/v1/text/contents", response_model=ContentsAnalysisResponse)
    async def contents_synthid(req: ContentAnalysisHttpRequest, request: Request) -> ContentsAnalysisResponse:
        results = await run_in_threadpool(
            _analyze_contents_sync, req.contents, req.detector_params, "synthid", request.app.state
        )
        return ContentsAnalysisResponse(root=results)

    @app.post("/v1/watermark/detect")
    async def watermark_detect(req: DetectRequest, request: Request) -> Dict[str, Any]:
        state = request.app.state
        settings: Settings = state.settings
        scheme = req.scheme or settings.default_scheme

        header_validation_id = request.headers.get("x-watermark-validation-id")
        if header_validation_id is not None:
            if not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                header_validation_id,
            ):
                raise HTTPException(
                    status_code=422,
                    detail="X-Watermark-Validation-Id must be a canonical lowercase UUID string",
                )
            try:
                header_validation_id = str(UUID(header_validation_id))
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="X-Watermark-Validation-Id must be a canonical lowercase UUID string",
                ) from exc
        if req.validation_id is not None and header_validation_id is not None:
            if req.validation_id != header_validation_id:
                raise HTTPException(
                    status_code=422,
                    detail="validation_id body/header values must match",
                )
        validation_id = req.validation_id or header_validation_id
        if req.texts is not None and (validation_id is not None or req.response_id is not None):
            raise HTTPException(
                status_code=422,
                detail="validation_id and response_id are supported only with a single 'text'",
            )

        if not state.keys:
            raise HTTPException(status_code=503, detail="no watermark keys configured")
        try:
            key = _resolve_key(req.key_id, state.keys, state.default_key)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if req.text is not None:
            payload: Dict[str, Any] = await run_in_threadpool(
                _build_detect_result, req.text, scheme, key, settings, state, validation_id, req.response_id
            )
        else:
            texts = req.texts or []

            def _build_all() -> List[Dict[str, Any]]:
                return [
                    _build_detect_result(t, scheme, key, settings, state, index=i)
                    for i, t in enumerate(texts)
                ]

            results = await run_in_threadpool(_build_all)
            payload = {"results": results}

        signature, signing = _sign_payload(payload, state.signing_key, state.signing_alg, settings.signing_key_id)
        return {**payload, "signature": signature, "signing": signing}

    return app


app = create_app()
