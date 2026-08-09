# TrustyAI/FMS detector contract and executed orchestrator wiring

This note distinguishes source-level API analysis from the version that was
actually deployed.

| Subject | Status | Source |
|---|---|---|
| TrustyAI detector contract | `OFFICIAL-SRC` / `STATIC` at reviewed `main` commit `747a4d3ef6f7d384b73f929a0162228ad56d98de` | [pinned source](https://github.com/trustyai-explainability/guardrails-detectors/tree/747a4d3ef6f7d384b73f929a0162228ad56d98de) |
| FMS Orchestrator 0.18.3 source behavior | `OFFICIAL-SRC` / `STATIC` at commit `6d2cec987223335adcc3803f884dae7a4aa59492`; not the deployed version | [pinned source](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/tree/6d2cec987223335adcc3803f884dae7a4aa59492) |
| Standalone FMS Phase 3 path | `EXECUTED` with orchestrator 0.16.0 on plain OpenShift | [fact D5](facts.md), [Phase 3 evidence](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half) |
| RHOAI-managed guardrails path | `EXECUTED` for the current internal metadata-only broker path; external KServe/Istio pass-through and supportability remain `OPEN` | [current managed-path/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted), [facts C11/D5](facts.md) |

The committed deployment pins
`quay.io/trustyai/ta-guardrails-orchestrator@sha256:f3952b137e2a21cb4445073d0d2ec6a76843ab3a957ecb8215158aa908027e1c`
(`STATIC`; [manifest](../deploy/phase3/orchestrator.yaml)). The executed
deployment used that recorded digest prefix, and its health endpoint reported
`fms-guardrails-orchestr8` 0.16.0 (`EXECUTED`; [append-only
correction](../EXPERIMENTS.md#corrections-to-earlier-entries-append-only-earlier-text-stands-as-history)).
The earlier `odh-3.4.2` image attribution was a research-candidate mix-up
(`STATIC`; pinned manifest and append-only correction above).
Version 0.18.3 is used below only as a pinned source-analysis baseline; it
was not the deployed orchestrator.

## Detector service contract

The TrustyAI detector contract uses `POST /api/v1/text/contents` with this
shape (`OFFICIAL-SRC`; pinned
[`scheme.py`](https://github.com/trustyai-explainability/guardrails-detectors/blob/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/common/scheme.py)):

```json
{
  "contents": ["text to inspect"],
  "detector_params": {}
}
```

The response is a list of lists whose outer length and order match `contents`.
An empty inner list means no detection; a positive inner list contains one or
more detection objects
(`OFFICIAL-SRC` / `STATIC`; pinned schema and handlers). The contract's
positive object includes content offsets, the detected text, detection type,
score, and optional metadata. Because positive responses contain the submitted
text, this interface is not a zero-retention or zero-disclosure boundary
(`STATIC`; schema). Application and platform handling must be assessed
separately (`OPEN`).

The detector error envelope is `{code, message}`, distinct from the
orchestrator client-facing `{details: ...}` envelope described below. The HTTP
request body is JSON (`Content-Type: application/json`) (`OFFICIAL-SRC`;
[vendored detector OpenAPI](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/docs/api/openapi_detector_api.yaml)).

The repository detector implements this contract plus a direct
`POST /v1/watermark/detect` endpoint (`STATIC`; [detector documentation](../detector/README.md)).
Both KGW and SynthID instances were built, deployed, and called through the
orchestrator (`EXECUTED`; fact D5 and the Phase 3 evidence above).

## Version-pinned static integration details

At the reviewed detector commit, `detector_params` is declared as
`Optional[Dict]` without a default. Under the repository's pinned Pydantic 2
dependency, omission is therefore expected to produce a validation error;
that omission case was not executed (`STATIC`; pinned
[`scheme.py`](https://github.com/trustyai-explainability/guardrails-detectors/blob/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/common/scheme.py)
and [`pyproject.toml`](https://github.com/trustyai-explainability/guardrails-detectors/blob/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/pyproject.toml)). Send an explicit
empty object when no detector parameters are needed.

The orchestrator's vendored detector OpenAPI marks `detector-id` as a
required header, while the two reviewed Python reference handlers do not read
that header (`OFFICIAL-SRC` / `STATIC`; [vendored
OpenAPI](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/docs/api/openapi_detector_api.yaml)
and [reference
handler](https://github.com/trustyai-explainability/guardrails-detectors/blob/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/huggingface/app.py)).
The 0.18.3 client source sets `detector-id`, `x-model-name`, and JSON content
type from the configured detector ID; other inbound headers are forwarded
only when allow-listed (`STATIC`; pinned
[`detector.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/clients/detector.rs)).

At the pinned 0.18.3 source, each detector entry requires `type`, `service`,
`chunker_id`, and `default_threshold`. The `whole_doc_chunker` sentinel needs
no separate chunker service and is accepted by
`/api/v2/text/detection/content` (`STATIC`; pinned
[`config.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/config.rs)
and [test
configuration](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/tests/test_config.yaml)).
For that endpoint, source removes the reserved `threshold` parameter for the
orchestrator's own score filtering and forwards the remaining detector
parameters downstream (`STATIC`; pinned
[`tasks.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/orchestrator/common/tasks.rs#L185-L235)).

The detector contract's error object uses `message`; the orchestrator's own
client-facing error uses `details`. In 0.18.3 source, downstream
400/404/422/503 statuses are passed through while other downstream statuses
become a masked 500 (`STATIC`; pinned
[`errors.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/server/errors.rs)).
Source also distinguishes `/health` liveness/version reporting from `/info`
dependency probes (`STATIC`; pinned
[`routes.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/server/routes.rs)); the observed
0.16.0 results are recorded separately below.
The reviewed detector implementations expose their own `GET /health` response
as the JSON string `"ok"` (`STATIC`; pinned
[`common/app.py`](https://github.com/trustyai-explainability/guardrails-detectors/blob/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/common/app.py)).

## Executed orchestrator request path

Phase 3 called:

```text
POST /api/v2/text/detection/content
```

with a string `content` and a map of detector IDs to parameter objects. The
orchestrator returned a flat `detections` list with `detector_id` attribution
(`EXECUTED`; [raw verdict matrix](../EXPERIMENTS.md#raw-evidence-phase-3-verdict-matrix-signing-retention-health-fresh-re-run)).

At the pinned 0.18.3 source, `content` and the detector map are both required
and non-empty; unknown request fields are denied. Successful results merge the
requested detector outputs into one list sorted by start offset, with the
orchestrator stamping `detector_id` on each result (`STATIC`; pinned
[`models.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/models.rs)
and [`detector.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/clients/detector.rs)).

The executed matrix established:

- KGW and SynthID samples fired only their matching detector;
- cross-scheme requests, unwatermarked model output, and human text returned
  no detection; and
- one request naming both detectors preserved the correct detector ID.

These are `EXECUTED` results for the recorded samples and configuration, not
general error rates or a production-readiness claim (fact D5).

The orchestrator's liveness and dependency information were exposed on the
dedicated port 8034. `/health` returned version 0.16.0 and `/info` reported
both detector services healthy (`EXECUTED`; [raw health output](../EXPERIMENTS.md#raw-evidence-phase-3-verdict-matrix-signing-retention-health-fresh-re-run)).

## Scheme authority and `path_prefix`

FMS Orchestrator 0.18.3 source contains `service.path_prefix`, which can
prefix downstream detector URLs (`STATIC`; [0.18.3 `config.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/0.18.3/src/config.rs)).
For example, `path_prefix: "/kgw"` composes the detector URL as
`/kgw/api/v1/text/contents`; separate detector entries can therefore select
KGW and SynthID service paths (`STATIC`; pinned 0.18.3 `config.rs` and
[`detector.rs`](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/clients/detector.rs)).
That is a source-level 0.18.3 option, not evidence that deployed 0.16.0 accepted
the field (`OPEN` for the deployed version).
The Phase 3 narrative reports that deployed 0.16.0 ignored that field, but
its command and raw output were not preserved, so that specific live claim is
`OPEN` ([Phase 3 narrative](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half)).
The repository uses separate KGW and SynthID Services/Deployments, each pinning
`WATERMARK_DETECTOR_SCHEME`, so detector identity—not client input—is the
accepted scheme authority (`STATIC` design + `EXECUTED` verdict matrix;
fact D5).

Version 0.18.3 source forwards detector parameters other than `threshold`
to the downstream service (`STATIC`; pinned `tasks.rs` above). The Phase 3
narrative also reports a deployed 0.16.0 `detector_params.scheme` override,
but no command/raw response was preserved, so that live observation remains
`OPEN`. Client overrides were deliberately omitted from the acceptance
matrix; empty parameter objects plus separate Services provided server-side
routing (`EXECUTED`; raw matrix above).

## Whole-document behavior

In 0.18.3 source, each detector configuration requires a `chunker_id`, and
the built-in sentinel `whole_doc_chunker` avoids a separate chunker service
for `/api/v2/text/detection/content` (`STATIC`; [0.18.3 source](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/tree/0.18.3/src)).
The committed Phase 3 configuration uses that whole-document mode
(`STATIC`; [orchestrator manifest](../deploy/phase3/orchestrator.yaml)), and
the recorded requests completed (`EXECUTED`; Phase 3 matrix).

## RHOAI lifecycle and managed-path boundary

RHOAI 3.4 documentation labels FMS Guardrails legacy, says it will be
deprecated in a future release, and directs users to NeMo Guardrails
(`OFFICIAL-SRC`; [fact C11](facts.md), [RHOAI documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/enabling_ai_safety_with_guardrails/enabling-ai-safety-with-guardrails_safety)).
The standalone 0.16.0 execution therefore establishes the recorded success
path against the historical detector contract, not the contract's complete
error behavior or a recommended or supported future RHOAI architecture.

Source documentation for `GuardrailsOrchestrator` shows a Kubernetes CR and
ConfigMap deployment shape (`OFFICIAL-SRC`; [TrustyAI tutorial](https://trustyai.org/docs/main/gorch-tutorial)).
Phase 3 did not deploy that RHOAI/operator-managed CR; that historical Phase 3
scope remains `OPEN` for that run (`STATIC`/`OPEN`; facts C11/D5). The subsequent
2026-08-09 run executed the current RHOAI-managed NeMo action with an
authenticated metadata-only broker and the synchronous one-in-`N` gateway
contract (`EXECUTED`; [current managed-path/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
External KServe/Istio pass-through, multi-replica/global sampling,
streaming/asynchronous behavior, platform-wide retention, and supportability
remain `OPEN` (same evidence; facts C8/D5/D6/D10).

## Detector validation and retention limits

The preserved negative probe showed that `load_settings()` accepted a NaN
threshold, zero SynthID depth, zero SynthID n-gram length, and KGW gamma 2
(`EXECUTED`, fact B23). The first remediation rejected those values and ran the
live D10 matrix, but independent reviews then found several missing upper
bounds and an explicit-blank vocabulary bypass. The current immutable detector
image matched local source, rejected both blank forms, accepted 9/9 exact maxima,
rejected 9/9 overflows, failed a controlled blank-valued rollout before readiness,
recovered, and completed the full D10 fixed matrix (`EXECUTED`; [current detector
reconciliation](../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09);
[current build-5 D10 rerun](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
Request-size/cache limits, expensive maximum combinations, and generation-side
bound parity remain `OPEN` (fact D9). Healthy 0.16.0 orchestrator dependencies
do not by themselves close detector-configuration or resource-hardening gaps.

Finite Phase 3 log-window checks found none of the tested distinctive sample
substrings in detector, SynthID-detector, or orchestrator logs (`EXECUTED`,
scoped; fact D5). The
detector code is stateless and hashes content in its own logs (`STATIC`). These
facts do not prove absolute or platform-wide zero retention, and the FMS
positive response itself echoes detected text (`STATIC`; contract above).

## Primary and local sources

- [TrustyAI detector source at the reviewed commit](https://github.com/trustyai-explainability/guardrails-detectors/tree/747a4d3ef6f7d384b73f929a0162228ad56d98de)
- [FMS Orchestrator v0.18.3 source at resolved commit](https://github.com/foundation-model-stack/fms-guardrails-orchestrator/tree/6d2cec987223335adcc3803f884dae7a4aa59492)
- [RHOAI 3.4 FMS lifecycle documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/enabling_ai_safety_with_guardrails/enabling-ai-safety-with-guardrails_safety)
- [Executed Phase 3 runbook](../deploy/phase3/README.md)
- [Fact register](facts.md) and [append-only evidence](../EXPERIMENTS.md)
