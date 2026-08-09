# vllm-watermark detector service

Phase 3 of `docs/implementation.md`: a standalone HTTP service that wraps the
**existing** watermark detectors in `src/vllm_watermark` (imported, not reimplemented:
`vllm_watermark.kgw.detector`, `vllm_watermark.synthid.detector`, `vllm_watermark.keys`,
`vllm_watermark.synthid.core`) behind two surfaces:

1. The TrustyAI/FMS Guardrails Orchestrator "detectors API" contract
   (`POST /api/v1/text/contents`, `GET /health`) — see "TrustyAI contract" below.
2. A direct `POST /v1/watermark/detect` endpoint with an optional detached-JWS signature.

The service import graph excludes `vllm` (`STATIC`; `detector/app.py`) — it imports only the vllm-free
`vllm_watermark.keys` / `vllm_watermark.kgw.core` / `vllm_watermark.kgw.detector` /
`vllm_watermark.synthid.core` / `vllm_watermark.synthid.detector` / `vllm_watermark.request_args`
modules, deliberately never `vllm_watermark.kgw.processor` / `vllm_watermark.synthid.processor`
(the two processor modules that import `vllm`). Historical compile and test
runs passed at their recorded revisions (`EXECUTED`; [run log](../EXPERIMENTS.md#2026-08-08--independent-post-push-review-correction)); this is not a claim about an untested later tree or the workstation's complete installed-package set.

## Why this integration path, and why it might not be the final one

`docs/facts.md` C5 registers the TrustyAI/FMS Guardrails Orchestrator's documented
`POST /api/v1/text/contents` detector contract. But `docs/facts.md` C11/D5 also record, `OFFICIAL-SRC`,
that RHOAI 3.4 documentation labels FMS Guardrails **legacy** and directs users to NeMo
Guardrails. The upstream NeMo 0.23.0 custom-action path was executed successfully; the
current RHOAI-managed `NemoGuardrails` CR and metadata-only broker path then
executed in a bounded one-replica synchronous, non-streaming D10 run
(`EXECUTED`; [current Phase 4/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
External KServe/Istio pass-through, supportability, key lifecycle, multi-replica
sampling, streaming/asynchronous behavior, and platform-wide retention remain
`OPEN` (same evidence; facts C4/C8/D4/D6/D10). This service targets the FMS contract because it is a real,
fetched, well-defined API surface, not because the legacy RHOAI packaging is the
recommended long-term production path.

## TrustyAI contract

Schema and route conventions are `OFFICIAL-SRC`, fetched directly from
`trustyai-explainability/guardrails-detectors` (Apache-2.0), commit
`747a4d3ef6f7d384b73f929a0162228ad56d98de` (`main`, fetched 2026-08-08 via
the [pinned source](https://github.com/trustyai-explainability/guardrails-detectors/tree/747a4d3ef6f7d384b73f929a0162228ad56d98de)):

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

Scheme selection:

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

Single-`text` response (flat object). These values are from the preserved
single-instance KGW Phase 3 capture (`EXECUTED`; [raw response](../EXPERIMENTS.md#raw-evidence-phase-3-verdict-matrix-signing-retention-health-fresh-re-run)); the JWS is elided:

```json
{
  "scheme": "kgw",
  "key_id": "poc-2026-08",
  "verdict": true,
  "z_score": 6.400354600105544,
  "p_value": 7.750825489048927e-11,
  "score": 0.9999999999224918,
  "num_tokens_scored": 188,
  "detector_version": "vllm-watermark-detector/0.1.0.dev0",
  "model_tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
  "scheme_details": {"num_green": 85, "gamma": 0.25},
  "signature": "<detached Ed25519 JWS elided>",
  "signing": "enabled"
}
```

`scheme_details` carries scheme-specific extras: `{num_green, gamma}` for `kgw`;
`{mean_g, score, depth, scorer}` for `synthid`. There, `score` is the selected
scorer's statistic: the weighted value for `weighted_mean`, or the same value
as `mean_g` for `mean`; `mean_g` always exposes the unweighted mean
(`STATIC`; `synthid/detector.py`).

`texts` (batch) response: `{"results": [<one flat object per text, in order>, ...], "signature":
..., "signing": ...}` — a documented design extension beyond the single-object shape,
also covered in `app.py`'s `DetectRequest` docstring. The whole batch is atomic: if
any text is too short to score for the requested scheme, the whole request 422s (naming which
`texts[i]` failed), rather than inventing a per-item partial-failure schema.

**Score mapping**: `score = clamp(1 - p_value, 0, 1)` is a monotone
rescaling, not a calibrated Bayesian probability (`STATIC`; `app.py`).

**Insufficient content**: a text with too few tokens to score at all (KGW: < 2 tokens; SynthID:
< `ngram_len` tokens, or every window's context was masked as repeated) is a `422` on
`/v1/watermark/detect` (a clear client-facing error for a purpose-built API) but an **empty
detection list** — not an error — on `/api/v1/text/contents` (consistent with that endpoint's
"no detection → `[]`" convention; a too-short string simply isn't flagged, matching how a
below-threshold string isn't flagged either).

### `GET /health`, `GET /ready`

- `/health` — always `{"status": "ok"}` once the process is up (liveness).
- `/ready` — `200` once the tokenizer is loaded AND the configured default watermark key is
  resolvable; `503` otherwise (`STATIC`; current route implementation).

The `/ready` handler itself checks tokenizer/default-key state, but lifespan
validates numeric configuration before the application can serve that route
(`STATIC`; `app.py`). The preserved negative probe found an earlier revision
accepting NaN/out-of-domain values, and later reviews found missing upper bounds
and an explicit-blank vocabulary bypass. The current immutable rebuild rejected
both blank forms and all nine overflows, accepted all nine exact maxima, failed a
controlled blank-valued rollout before readiness, recovered, and completed the
full D10 fixed matrix (`EXECUTED`; [current detector
reconciliation](../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09);
[current build-5 D10 rerun](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
Request-size/cache limits and generation-processor bound parity remain `OPEN`
(fact D9 and the same review record).

## JWS signing of `/v1/watermark/detect` responses

If `SIGNING_KEY_PATH` points to a PEM-encoded RSA or Ed25519 **private** key, every
successful `/v1/watermark/detect` detection response carries a **detached** JWS
(RFC 7797, `b64: false`
unencoded-payload mode) over the canonical JSON encoding of the response payload **excluding**
the `signature`/`signing` fields (`STATIC`; `app.py`):

```python
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
```

The algorithm (`RS256` for RSA, `EdDSA` for Ed25519) is auto-detected from the loaded key's
Python type at startup — never configured separately, so it cannot drift from what the key
actually is. `SIGNING_KEY_ID` (optional) becomes the JWS header's `kid` claim.

If `SIGNING_KEY_PATH` is unset, the current service starts in unsigned mode
with `"signature": null` and `"signing": "disabled"` (`STATIC`; `app.py`).
Both executed detector Deployments require a signing Secret; unsigned mode is
not evidence of an accepted production policy. If the path is configured but
unreadable, unparsable, or an unsupported key type, startup fails (`STATIC`;
current implementation).

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

## Application-level content-handling scope

The application has no persistence layer or request-body logging middleware,
and its own detection log path emits a SHA-256 prefix plus verdict metadata
instead of submitted text (`STATIC`; `detector/app.py` and service tests).
The recorded service suite exercised content-leak assertions, and finite
Phase 3 log-window checks found none of the tested distinctive sample
substrings in detector, SynthID-detector, or orchestrator logs (`EXECUTED`, scoped; fact D5 and
[`EXPERIMENTS.md`](../EXPERIMENTS.md)).

This does **not** prove absolute or system-wide zero retention. Positive FMS
contract responses echo detected text (`STATIC`; [contract note](../docs/api-notes-trustyai-detectors.md)),
and the upstream NeMo path has separate 422/event-log content-handling gaps
that remain relevant to the managed deployment ([NeMo note](../docs/api-notes-nemo-guardrails.md)).
The current run's finite hash-only scan found no marker or secret matches on its
named surfaces (`EXECUTED` scoped), but end-to-end platform retention and
mitigation remain `OPEN` (same evidence; [fact D5](../docs/facts.md)); this
service note makes no legal conclusion.

## Threat notes (deployment policy, not enforced by this service)

- **Access control**: this service has **no built-in authentication/authorization**. Who may call
  `/api/v1/text/contents` / `/v1/watermark/detect` at all is a deployment-time policy decision
  (`STATIC`; `detector/app.py`). The service does not implement the access
  conditions quoted in [fact A10](../docs/facts.md); deployment-specific
  application of those conditions remains `OPEN`.
- **Signing key handling**: `SIGNING_KEY_PATH` should point to a mounted Secret (not baked into
  an image), matching the same handling rules as `WATERMARK_KEYS`/`WATERMARK_KEY` (AGENTS.md §3).
  The signing and watermark keys serve different implementation roles
  (`STATIC`; `app.py` and watermark core); generate, store, scope, and rotate
  them independently (`OPEN`; D4).
- **Detection is not proof of authorship**: a positive detection means "this text is statistically
  consistent with having been sampled under this key/scheme" — it is not, on its own, a legal
  attestation. Treat verdicts as one signal among several, per the same caution the rest of this
  repo applies to watermarking's known robustness limits (`docs/facts.md` B17).
- **`WATERMARK_VOCAB_SIZE` consistency**: the service does not compare its
  configured value with the generation deployment (`STATIC`; `app.py`). It
  must be pinned consistently; mismatch behavior is not quantified in the
  preserved evidence (`OPEN`).

## Configuration (env vars)

See `detector/app.py`'s module docstring for the complete, authoritative list (every var, its
default, and — for the vars shared with the generation-side plugins — *why* the name/default is
reused rather than independently chosen). Summary:

| Var | Default | Notes |
|---|---|---|
| `WATERMARK_KEYS` / `WATERMARK_KEY` (+ `WATERMARK_KEY_ID`) | *(none — service starts, `/ready` 503s, detection 503s)* | Reused unchanged from `vllm_watermark.keys` |
| `WATERMARK_DETECTOR_SCHEME` | `kgw` | Default scheme when not forwarded/forced |
| `MODEL_TOKENIZER` | `Qwen/Qwen2.5-0.5B-Instruct` | Pre-loaded once at startup |
| `WATERMARK_VOCAB_SIZE` | *(falls back to `len(tokenizer)`, with a loud warning)* | **Must** match generation-time vocab size exactly; inclusive range `1..1048576`, with an additional non-empty effective KGW-green-list check |
| `WATERMARK_Z_THRESHOLD` | `4.0` | Shared by both schemes; inclusive range `0..100` |
| `WATERMARK_KGW_IGNORE_REPEATED_NGRAMS` | `on` | Matches the Phase 1 measured configuration |
| `WATERMARK_SYNTHID_SCORER` | `weighted_mean` | Matches `docs/implementation.md` Phase 2 guidance |
| `VLLM_WATERMARK_GAMMA`, `VLLM_WATERMARK_DELTA` | `0.25`, `2.0` | **Reused name+default** from `kgw/processor.py`; gamma is in `(0,1)`, delta in inclusive `0..100` |
| `VLLM_WATERMARK_SYNTHID_NGRAM_LEN` | `5` | **Reused name+default** from `synthid/processor.py`; inclusive range `1..1024` |
| `VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SIZE` | `65536` | Inclusive range `1..16777216` |
| `VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SEED` | `0` | Inclusive PyTorch seed range `-2**63..2**64-1` |
| `VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE` | `1024` | Inclusive range `0..65536` |
| `VLLM_WATERMARK_SYNTHID_KEY_DEPTH` | `30` (`DEFAULT_SYNTHID_DEPTH`) | Inclusive range `1..256`; subkeys derived with `SYNTHID_KEY_LABEL` |
| `SIGNING_KEY_PATH`, `SIGNING_KEY_ID` | *(unset → unsigned)* | See "JWS signing" above |

The table describes the current source defaults and service-side validation
policy (`STATIC`; `detector/app.py`). Exact maxima and overflows were exercised
both through local lifespan and inside the deployed image; one real invalid
rollout and recovery were also executed (`EXECUTED`; [upper-bound
record](../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09)).
These per-setting bounds are not a claim that arbitrary request sizes or every
maximum combination are production-safe; those boundaries remain `OPEN` under
fact D9.

## Running locally

```bash
pip install --user -r detector/requirements.txt
# vllm_watermark itself: either add src/ to PYTHONPATH (see detector/tests/conftest.py for the
# exact pattern), or `pip install --user dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl`
# (pure-python wheel, does not pull vllm).

read -r -s -p 'WATERMARK_KEY (hex): ' WATERMARK_KEY
export WATERMARK_KEY
printf '\n'
export MODEL_TOKENIZER=Qwen/Qwen2.5-0.5B-Instruct
export WATERMARK_VOCAB_SIZE=151936   # Qwen2.5-0.5B-Instruct's model_config.vocab_size

repo_root=$(pwd)                      # run from the repository root
export PYTHONPATH="$repo_root/src"
cd detector && uvicorn app:app --host 0.0.0.0 --port 8080
```

## Container build

The executed container definition is [`detector/Dockerfile`](Dockerfile); the on-cluster binary
build and deployment commands are in [`deploy/phase3/README.md`](../deploy/phase3/README.md). The recorded image
build installs the wheel, CPU-only torch, and pinned detector dependencies and serves
on port 8000 (`EXECUTED`; Phase 3 run in [`EXPERIMENTS.md`](../EXPERIMENTS.md)).

## Tests

```bash
/usr/bin/python3 -m pytest -q
```

At the recorded audit revision, that exact combined command passed 154 tests
(`EXECUTED`; ["Combined test-suite invocation" raw output in the append-only evidence log](../EXPERIMENTS.md)).
The current upper-bound revision passed 129 detector service tests, including
81 focused startup cases (`EXECUTED`; [raw
output](../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09)).
All test results are revision-scoped; rerun after changes.
