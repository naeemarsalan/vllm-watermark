# vllm-watermark detector service

Task B3 (Phase 3, `docs/implementation.md`). A standalone HTTP service that wraps the
**existing** watermark detectors in `src/vllm_watermark` (imported, not reimplemented:
`vllm_watermark.kgw.detector`, `vllm_watermark.synthid.detector`, `vllm_watermark.keys`,
`vllm_watermark.synthid.core`) behind two surfaces:

1. The TrustyAI/FMS Guardrails Orchestrator "detectors API" contract
   (`POST /api/v1/text/contents`, `GET /health`) — see "TrustyAI contract" below.
2. A direct `POST /v1/watermark/detect` endpoint with an optional detached-JWS signature.

This service **never imports `vllm`** — it imports only the vllm-free
`vllm_watermark.keys` / `vllm_watermark.kgw.core` / `vllm_watermark.kgw.detector` /
`vllm_watermark.synthid.core` / `vllm_watermark.synthid.detector` / `vllm_watermark.request_args`
modules, deliberately never `vllm_watermark.kgw.processor` / `vllm_watermark.synthid.processor`
(the only two modules in the package that import `vllm`). Verified: `detector/app.py`
`py_compile`s and the full local test suite (below) passes with no `vllm` installed anywhere
on this workstation.

## Why this integration path, and why it might not be the final one

`docs/facts.md` C5 confirms the TrustyAI/FMS Guardrails Orchestrator accepts any detector
implementing `POST /api/v1/text/contents`. But `docs/facts.md` C11/D5 also record, `OFFICIAL-SRC`,
that RHOAI 3.4 documentation labels FMS Guardrails **legacy** and directs users to NeMo
Guardrails — whether/how NeMo's extension surface can call a detector like this one is still
`OPEN` (D5) as of this task. This service targets the FMS contract because it is a real,
fetched, well-defined API surface today (see citation below) — not a claim that it is the
recommended production integration path going forward. Re-verify against the current NeMo
Guardrails extension surface before treating this as the final integration.

## TrustyAI contract — fetched, not from memory

Schema and route conventions were fetched directly from
`trustyai-explainability/guardrails-detectors` (Apache-2.0), commit
`747a4d3ef6f7d384b73f929a0162228ad56d98de` (`main`, fetched 2026-08-08 via
`gh api repos/trustyai-explainability/guardrails-detectors/...`):

- `detectors/common/scheme.py` — the exact pydantic field names/types this service ported
  verbatim (`ContentAnalysisHttpRequest`, `ContentAnalysisResponse`, `ContentsAnalysisResponse`,
  `Error`, `EvidenceObj`).
- `detectors/common/app.py` — `GET /health` → bare string `"ok"`; 422 handler shape
  `{"code": 422, "message": "..."}`.
- `detectors/huggingface/app.py` — route registration (`POST /api/v1/text/contents`,
  `response_model=ContentsAnalysisResponse`) and the `run_in_threadpool` convention (CPU-bound
  detector work kept off the event loop — reused here).
- `detectors/huggingface/detector.py` — the "no detection → `[]`" convention (`run()` appends a
  possibly-empty list per content string), and the "read `detector_params` defensively, warn +
  fall back rather than 422" convention — mirrored by this service's own `detector_params.scheme`
  / `detector_params.key_id` handling.

Full citations, including raw.githubusercontent.com URLs for each file, are in `detector/app.py`'s
module docstring.

## Endpoints

### `POST /api/v1/text/contents` (+ scheme-forced aliases)

TrustyAI-contract endpoint. Body: `{"contents": ["text1", "text2", ...], "detector_params": {...}}`.
Response: one list per submitted content, **in order** — an empty list means "not detected as
watermarked"; a positive detection is one `ContentAnalysisResponse` spanning the whole content
(`start=0, end=len(content)`).

Scheme selection (per Task B3 spec):

1. `detector_params.scheme` (`"kgw"` | `"synthid"`), if the orchestrator forwards `detector_params`.
2. Else env `WATERMARK_DETECTOR_SCHEME` (default `"kgw"`).

`detector_params.key_id` selects which configured watermark key to use, same fallback logic
(else the configured default key). Both are read **defensively** (TrustyAI convention, see
above): an invalid `scheme` value logs a warning and falls back to the default rather than 422ing
the whole request; an invalid/unresolvable `key_id` is a `400`.

Two scheme-forced alias routes are also exposed — cheap insurance for an orchestrator/guardrails
configuration that registers one detector endpoint per behavior and cannot forward
`detector_params` through at all:

- `POST /kgw/api/v1/text/contents` — identical handler, scheme **forced** to `"kgw"` (an explicit
  `detector_params.scheme` is ignored in favor of the route; `detector_params.key_id` still works).
- `POST /synthid/api/v1/text/contents` — same, forced to `"synthid"`.

### `POST /v1/watermark/detect`

Direct endpoint. Body: exactly one of `text` (string) or `texts` (non-empty list of strings),
plus optional `key_id` and `scheme` (`"kgw"` | `"synthid"`, strictly validated — an invalid value
is a `422`, unlike the TrustyAI route's defensive fallback: this is a purpose-built typed API, not
an orchestrator-forwarded params bag).

Single-`text` response (flat object):

```json
{
  "scheme": "kgw",
  "key_id": "default",
  "verdict": true,
  "z_score": 21.35,
  "p_value": 3.2e-101,
  "score": 1.0,
  "num_tokens_scored": 255,
  "detector_version": "vllm-watermark-detector/0.1.0.dev0",
  "model_tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
  "scheme_details": {"num_green": 213, "gamma": 0.25},
  "signature": "eyJhbGciOiJSUzI1NiIs...b64false...header..<sig>",
  "signing": "enabled"
}
```

`scheme_details` carries scheme-specific extras: `{num_green, gamma}` for `kgw`;
`{mean_g, score, depth, scorer}` for `synthid` (`score` there is the *weighted* value the
z-score/verdict were actually computed from — `mean_g` is always the unweighted mean, for
comparability regardless of `scorer`).

`texts` (batch) response: `{"results": [<one flat object per text, in order>, ...], "signature":
..., "signing": ...}` — a design extension beyond the single-object shape in the Task B3 line
item, documented here and in `app.py`'s `DetectRequest` docstring. The whole batch is atomic: if
any text is too short to score for the requested scheme, the whole request 422s (naming which
`texts[i]` failed), rather than inventing a per-item partial-failure schema.

**Calibrated `score`**: `score = clamp(1 - p_value, 0, 1)`, documented in full (including why it
is a monotone rescaling, not a Bayesian probability) in `app.py`'s module docstring "Calibrated
`score` mapping".

**Insufficient content**: a text with too few tokens to score at all (KGW: < 2 tokens; SynthID:
< `ngram_len` tokens, or every window's context was masked as repeated) is a `422` on
`/v1/watermark/detect` (a clear client-facing error for a purpose-built API) but an **empty
detection list** — not an error — on `/api/v1/text/contents` (consistent with that endpoint's
"no detection → `[]`" convention; a too-short string simply isn't flagged, matching how a
below-threshold string isn't flagged either).

### `GET /health`, `GET /ready`

- `/health` — always `{"status": "ok"}` once the process is up (liveness).
- `/ready` — `200` once the tokenizer is loaded AND at least one watermark key is configured;
  `503` otherwise (readiness — gates traffic on the service actually being able to detect
  anything, without taking the process down over a config problem `/health` shouldn't reflect).

## JWS signing of `/v1/watermark/detect` responses

If `SIGNING_KEY_PATH` points to a PEM-encoded RSA or Ed25519 **private** key, every
`/v1/watermark/detect` response carries a **detached** JWS (RFC 7797, `b64: false`
unencoded-payload mode) over the canonical JSON encoding of the response payload **excluding**
the `signature`/`signing` fields:

```python
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
```

The algorithm (`RS256` for RSA, `EdDSA` for Ed25519) is auto-detected from the loaded key's
Python type at startup — never configured separately, so it cannot drift from what the key
actually is. `SIGNING_KEY_ID` (optional) becomes the JWS header's `kid` claim.

If `SIGNING_KEY_PATH` is **unset**, the service starts and serves completely normally —
**never fail closed on a missing signing key in dev** — with every response's `"signature": null`
and `"signing": "disabled"`. If it **is** set but unreadable/unparsable/an unsupported key type,
startup fails loudly (a broken, as opposed to absent, signing key is a deployment bug worth
surfacing immediately).

To verify a signed response (PyJWT — the library this service itself uses, executed and verified
working end-to-end in `detector/tests/test_service.py::TestSigning`):

```python
import json, jwt

body = response.json()
signature = body.pop("signature")
body.pop("signing")
canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

decoded = jwt.api_jws.decode_complete(
    signature, key=public_key, algorithms=["RS256"],  # or ["EdDSA"]
    detached_payload=canonical,
)
# decoded["header"] == {"alg": "RS256", "b64": False, "crit": ["b64"], "typ": "JOSE", "kid": "..."}
```

## Zero retention (Code of Practice measure)

Submitted text is **never logged, never persisted, never embedded in an exception message**. The
only content-derived value ever logged is `sha256(content.encode("utf-8")).hexdigest()[:16]`,
alongside the resolved scheme/key_id/verdict/latency. This service adds **no** request-body
logging middleware — `detector/tests/test_service.py::TestZeroRetention::test_no_body_logging_middleware_installed`
asserts `app.user_middleware == []`, and
`test_content_never_appears_in_captured_logs` / `test_insufficient_tokens_error_never_embeds_text`
assert the submitted text (whole string, a 40-char prefix, and the too-short-text error path)
never appears anywhere in `caplog`'s captured records, across both endpoints.

`docs/facts.md` A9 (`OJ-VERBATIM`) records that logging/fingerprinting alone is insufficient and a
detection mechanism must be *available* for Art. 50(2) compliance; `docs/implementation.md`
Phase 3 calls for "zero retention of submitted content... log only hashes plus verdicts" as an
engineering design goal — this service implements that design goal. Whether that specific
zero-retention shape is itself a **Code of Practice** *requirement* (as opposed to sound
engineering practice this task adopts) remains `OPEN` per `docs/implementation.md` Phase 3 ("Verify
and register the exact source before describing zero retention as a Code requirement") — this
service does not claim otherwise.

## Threat notes (deployment policy, not enforced by this service)

- **Access control**: this service has **no built-in authentication/authorization**. Who may call
  `/api/v1/text/contents` / `/v1/watermark/detect` at all is a deployment-time policy decision
  (network policy, a service mesh mTLS boundary, an API gateway in front of it, etc.) — `A10`
  (`OJ-VERBATIM`, `docs/facts.md`) notes the Code permits professional-setting deployments to
  restrict detector access to affected persons; this service does not implement that restriction
  itself, it assumes the deployment's ingress/network layer does.
- **Signing key handling**: `SIGNING_KEY_PATH` should point to a mounted Secret (not baked into
  an image), matching the same handling rules as `WATERMARK_KEYS`/`WATERMARK_KEY` (AGENTS.md §3).
  The signing key is a *response-integrity* key (proves "this detector produced this verdict"),
  entirely separate key material from the watermark key(s) themselves — compromising one does not
  compromise the other.
- **Detection is not proof of authorship**: a positive detection means "this text is statistically
  consistent with having been sampled under this key/scheme" — it is not, on its own, a legal
  attestation. Treat verdicts as one signal among several, per the same caution the rest of this
  repo applies to watermarking's known robustness limits (`docs/facts.md` B17).
- **`WATERMARK_VOCAB_SIZE` misconfiguration is a silent failure mode**: if it doesn't exactly
  match the model's generation-time vocab_size, scores land near zero with **no error raised** —
  see `app.py`'s module docstring. There is no way for this service to detect that mismatch on its
  own; it must be operationally pinned to the same value used at generation time.

## Configuration (env vars)

See `detector/app.py`'s module docstring for the complete, authoritative list (every var, its
default, and — for the vars shared with the generation-side plugins — *why* the name/default is
reused rather than independently chosen). Summary:

| Var | Default | Notes |
|---|---|---|
| `WATERMARK_KEYS` / `WATERMARK_KEY` (+ `WATERMARK_KEY_ID`) | *(none — service starts, `/ready` 503s, detection 503s)* | Reused unchanged from `vllm_watermark.keys` |
| `WATERMARK_DETECTOR_SCHEME` | `kgw` | Default scheme when not forwarded/forced |
| `MODEL_TOKENIZER` | `Qwen/Qwen2.5-0.5B-Instruct` | Pre-loaded once at startup |
| `WATERMARK_VOCAB_SIZE` | *(falls back to `len(tokenizer)`, with a loud warning)* | **Must** match generation-time vocab_size exactly |
| `WATERMARK_Z_THRESHOLD` | `4.0` | Shared by both schemes |
| `WATERMARK_KGW_IGNORE_REPEATED_NGRAMS` | `on` | Matches the Phase 1 measured configuration |
| `WATERMARK_SYNTHID_SCORER` | `weighted_mean` | Matches `docs/implementation.md` Phase 2 guidance |
| `VLLM_WATERMARK_GAMMA`, `VLLM_WATERMARK_DELTA` | `0.25`, `2.0` | **Reused name+default** from `kgw/processor.py` |
| `VLLM_WATERMARK_SYNTHID_NGRAM_LEN` | `5` | **Reused name+default** from `synthid/processor.py` |
| `VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SIZE` | `65536` | ″ |
| `VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SEED` | `0` | ″ |
| `VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE` | `1024` | ″ |
| `VLLM_WATERMARK_SYNTHID_KEY_DEPTH` | `30` (`DEFAULT_SYNTHID_DEPTH`) | ″ — subkeys derived with `SYNTHID_KEY_LABEL` |
| `SIGNING_KEY_PATH`, `SIGNING_KEY_ID` | *(unset → unsigned)* | See "JWS signing" above |

## Running locally

```bash
pip install --user -r detector/requirements.txt
# vllm_watermark itself: either add src/ to PYTHONPATH (see detector/tests/conftest.py for the
# exact pattern), or `pip install --user dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl`
# (pure-python wheel, does not pull vllm).

export WATERMARK_KEY=<hex secret>
export MODEL_TOKENIZER=Qwen/Qwen2.5-0.5B-Instruct
export WATERMARK_VOCAB_SIZE=151936   # Qwen2.5-0.5B-Instruct's model_config.vocab_size

cd detector && uvicorn app:app --host 0.0.0.0 --port 8080
```

## Container build

Per Task B3: "its container installs the vllm-watermark wheel + detector deps". Sketch (not
executed on this task — no cluster write access; see AGENTS.md §Cluster):

```dockerfile
FROM registry.access.redhat.com/ubi9/python-312:latest
COPY dist/vllm_watermark-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/vllm_watermark-*.whl
COPY detector/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu
COPY detector/app.py /app/app.py
WORKDIR /app
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Tests

```bash
/usr/bin/python3 -m pytest detector/tests/test_service.py -v
```

30 tests, run locally (no vLLM, no GPU, `MODEL_TOKENIZER=gpt2` — see
`detector/tests/test_service.py` module docstring for why gpt2 is the right choice for a
self-consistency test suite). The raw pytest evidence for this suite lives in
`EXPERIMENTS.md` (the combined-suite run `154 passed` with its exact invocation, plus the audit
addendum); re-run locally with `python3 -m pytest detector/tests -q`. Historical note kept per the
repo's normal verification-logging convention.
