# NeMo Guardrails extension surface and RHOAI boundary

This note separates three different guardrails paths. They are not
interchangeable, and evidence for one must not be used to claim another
(`STATIC`; the scoped status/evidence table below).

| Path | Current status | Evidence |
|---|---|---|
| Standalone FMS Guardrails Orchestrator | `EXECUTED`, but RHOAI 3.4 labels the product path legacy | [FMS contract note](api-notes-trustyai-detectors.md), [fact C11](facts.md), [Phase 3 run](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half) |
| Upstream `nemoguardrails==0.23.0` custom action | `EXECUTED` as a standalone upstream-library PoC | [fact D5](facts.md); "NeMo Guardrails forward-path validation" and "NeMo PoC hardening evidence" in the [append-only evidence log](../EXPERIMENTS.md) |
| RHOAI-managed `NemoGuardrails` custom resource | `EXECUTED` for the current internal metadata-only broker path; external gateway/Istio pass-through, supportability, and platform-wide retention remain `OPEN` | [current managed-path/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted), [implementation status](implementation.md#phase-4--rhoai-deployment-pattern-scopes-c8-informs-d6) |

The upstream tag `v0.23.0` resolves to commit
`dc046e4e1db894893214ffab487c35f451f5baad` (`OFFICIAL-SRC`; [pinned
source](https://github.com/NVIDIA-NeMo/Guardrails/tree/dc046e4e1db894893214ffab487c35f451f5baad)).
Source claims in this note use that revision unless another scope is stated.

The executed upstream-library result is not evidence of Red Hat product
support, RHOAI operator behavior, or production readiness (`OPEN`; facts
C4/C11/D5/D6).

## Upstream extension mechanism

NeMo Guardrails supports custom Python actions that can be referenced by
input or output rail flows. An output action can receive the generated bot
message and call an external HTTP service (`OFFICIAL-SRC`; upstream
[`action parameters`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/docs/configure-rails/actions/action-parameters.mdx)
and [`creating actions`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/docs/configure-rails/actions/creating-actions.mdx)).
NeMo does not impose the FMS
`POST /api/v1/text/contents` detector contract on such actions; the action
defines its own request and response mapping (`STATIC`; upstream 0.23.0
source and the committed action below).

A minimal output-rail registration names the flow that invokes the action:

```yaml
rails:
  output:
    flows:
      - watermark check
```

The flow name and action result mapping are configuration-specific; this shape
does not itself establish a detector protocol or managed-RHOAI behavior
(`OFFICIAL-SRC` / `STATIC`; pinned upstream
[`output-rail example`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/examples/configs/self_check_thinking/config.yml)
and the [committed 0.23.0 flow](../deploy/phase3/nemo-guardrails-poc.yaml)).

The reusable repository artifact is
[`deploy/phase3/nemo-guardrails-poc.yaml`](../deploy/phase3/nemo-guardrails-poc.yaml),
which contains the rail configuration, flow, and hardened action. Its
version-resolved Python dependencies are in
[`nemo-poc-constraints.txt`](../deploy/phase3/nemo-poc-constraints.txt)
(`STATIC`; committed files). Historical `/tmp` helpers shown in the run log
were evidence-capture utilities in a deleted pod, not a committed replay
harness (`EXECUTED` historical fact; [documentation-boundary correction](../EXPERIMENTS.md#documentation-boundary-corrections-final-pass)).

### Version-specific integration details

At the pinned revision, actions placed in a configuration's `actions.py` or
`actions/` package are auto-registered when that configuration loads, and the
special `context` parameter exposes `bot_message` to output rails
(`OFFICIAL-SRC`; upstream [`registration`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/docs/configure-rails/actions/registering-actions.mdx)
and [`action parameters`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/docs/configure-rails/actions/action-parameters.mdx)).
The committed 0.23.0 PoC loads a local `actions.py`, receives
`context["bot_message"]`, calls the detector with `httpx`, and returns a
plain Boolean consumed by its Colang flow (`STATIC`; [committed
artifact](../deploy/phase3/nemo-guardrails-poc.yaml)). That registration and
the recorded block/pass flow were exercised (`EXECUTED`; "NeMo PoC
hardening evidence" in the [append-only evidence log](../EXPERIMENTS.md)).

The pinned 0.23.0 tree does not contain the later `RailOutcome` module used by
develop-branch examples (`STATIC`; pinned source tree). Those examples are not
compatible recipes for the executed package and are intentionally not
reproduced here. The PoC also did not use NeMo's optional actions server:
`actions_server_url` redirects action execution to a separate NeMo action-RPC
process; it is not a generic detector protocol (`OFFICIAL-SRC`; upstream
[`actions-server documentation`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/docs/run-rails/using-fastapi-server/actions-server.mdx)).
At this revision that RPC is `POST /v1/actions/run`, accepts `action_name` and
`action_parameters`, and returns `status` plus `result`; it dispatches an action
already registered in NeMo rather than accepting the FMS detector contract
(`STATIC`; pinned upstream
[`actions-server source`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/nemoguardrails/actions_server/actions_server.py)).

## What ran with upstream 0.23.0

The custom output action called the direct detector endpoint. A known KGW
sample was blocked and a human sample passed through both the Python API and
the server's `POST /v1/checks` endpoint (`EXECUTED`; [fact D5](facts.md),
[raw transcript](../EXPERIMENTS.md#7-real-detector-happy-path-regression-kgw--blocked-human--passed-via-v1checks)).

The executed `/v1/checks` body used a top-level `model`, user and assistant
messages, and `guardrails.config_id` (singular). The response was reduced to
status and rail fields in the preserved output so submitted text was not
reprinted (`EXECUTED`; same transcript). Assistant-only routing is supported
by upstream source (`STATIC`) but was not the executed request shape.

`POST /v1/checks` evaluates supplied messages through `check_async`; it does
not ask this endpoint to create a new chat completion (`OFFICIAL-SRC`; pinned
[`server route`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/nemoguardrails/server/api.py#L673-L732)).
The version-specific request shape is:

```json
{
  "model": "<required model identifier>",
  "messages": [
    {"role": "user", "content": "<omitted>"},
    {"role": "assistant", "content": "<omitted>"}
  ],
  "guardrails": {"config_id": "watermark-poc"}
}
```

With no explicit rail-type override, pinned source maps user-only messages to
input rails, assistant-only messages to output rails, and a user-plus-assistant
sequence to both (`STATIC`; pinned
[`check_async`](https://github.com/NVIDIA-NeMo/Guardrails/blob/dc046e4e1db894893214ffab487c35f451f5baad/nemoguardrails/rails/llm/llmrails.py#L1619-L1686)).

The hardening transcript also established the following narrow results:

- Missing-verdict and non-boolean-verdict HTTP-200 responses were blocked
  (`EXECUTED`; [items 3–4](../EXPERIMENTS.md#3-malformed-200-missing-verdict--blocked)).
- Five adversarial detector-returned marker strings occurred zero times in
  the captured 135-line process output; sanitized request-local metadata was
  logged (`EXECUTED`; [item 5](../EXPERIMENTS.md#5-adversarial-marker-fields-case-log-tunneling-defect-fix--all-marker-counts-0)).
- The fresh-pod package freeze matched all 78 entries in the committed
  constraints file (`EXECUTED`; [item 2](../EXPERIMENTS.md#2-pinned-install--pip-freeze-diff-zero-differences)).

The earlier upstream-NeMo action has a fail-closed detector-outage branch
(`STATIC`; committed action); its hardening transcript preserves malformed-response
execution, not a live outage. Separately, the current RHOAI-managed
metadata-only broker path executed a real detector outage: it exhausted three
attempts and the synchronous gateway returned a content-free 503 (`EXECUTED`;
[fact D5](facts.md), [current managed-path evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
That scoped result must not be broadened into platform-wide outage behavior, which
remains `OPEN`.

## RHOAI-managed `NemoGuardrails`: executed scope and open boundaries

The RHOAI 3.4 CR schema defines
`apiVersion: trustyai.opendatahub.io/v1alpha1`, kind `NemoGuardrails`, and a
`spec.nemoConfigs` list whose ConfigMaps are mounted under
`/app/config/<name>` (`OFFICIAL-SRC`; [exported RHOAI 3.4 schema](https://github.com/opendatahub-io/architecture-context/blob/main/architecture/rhoai-3.4/contracts/schemas/trustyai-service-operator/nemoguardrails.v1alpha1.json)).
The schema demonstrates the mount shape; source inspection alone does not prove
that a custom `actions.py` mounts, auto-registers, and behaves correctly in the
shipped operator image (`STATIC`). The current internal metadata-only broker
path did execute through the RHOAI-managed resource (`EXECUTED`; [current
managed-path/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

Source inspection of the dashboard provisioning path found a default
ConfigMap containing model, prompt, and rail files but no custom action
(`STATIC`; [dashboard source](https://github.com/opendatahub-io/odh-dashboard/blob/main/packages/gen-ai/bff/internal/integrations/kubernetes/nemo_guardrails.go)).
That observation describes the reviewed default only; it does not establish
a product limitation or a supported administrator customization path.

The current run established, in a bounded single-replica synchronous and
non-streaming scope (`EXECUTED`; [current managed-path/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)):

- the RHOAI-shipped NeMo version and image;
- custom-action ConfigMap mounting and registration through the managed CR;
- the real service/gateway request path;
- finite hash-only scans of the named validation, event, platform, and response
  surfaces; and
- metadata-only correlation through the managed action and authenticated broker;
- one-in-`N` selection with `N=1` and `N=5`, including retry, queue, metrics,
  detector-outage, and fail-closed checks.

The external KServe/Istio pass-through, product supportability, multi-replica or
global sampling, streaming/asynchronous behavior, and platform-wide retention
remain `OPEN` (`OPEN`; [current managed-path/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted), [fact register](facts.md)).

## Retention, failure, and detector-configuration limits

The detector application is stateless by design, and finite Phase 3
log-window checks found none of the tested distinctive sample substrings
(`STATIC` + scoped `EXECUTED`, fact D5). This is not an absolute
zero-retention result. Upstream NeMo 0.23.0 source inspection identified
request-body echo on 422 validation errors and message content in its event
log (`STATIC`; finding recorded in the
"NeMo Guardrails forward-path validation" section in the [append-only evidence log](../EXPERIMENTS.md)).
The current finite hash-only scan found no marker or secret matches on its named
surfaces, but platform-wide retention and mitigation remain `OPEN` (`EXECUTED`
scoped / `OPEN`; [current managed-path/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

Fail-closed transport/schema handling does not protect against a detector
with semantically invalid numeric configuration. The preserved negative
probe showed that `load_settings()` accepted a NaN threshold, zero SynthID
depth, zero SynthID n-gram length, and KGW gamma 2 (`EXECUTED`, fact B23);
source wiring adds no startup/readiness validation for them (`STATIC`).
Fail-fast validation, detector image rebuild, and the live detector matrix were
executed in the current bounded run (`EXECUTED`; [current managed-path/D10
evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

## Primary and local sources

- [NeMo Guardrails v0.23.0 source at resolved commit](https://github.com/NVIDIA-NeMo/Guardrails/tree/dc046e4e1db894893214ffab487c35f451f5baad)
- [RHOAI 3.4 `NemoGuardrails` schema](https://github.com/opendatahub-io/architecture-context/blob/main/architecture/rhoai-3.4/contracts/schemas/trustyai-service-operator/nemoguardrails.v1alpha1.json)
- [RHOAI dashboard provisioning source](https://github.com/opendatahub-io/odh-dashboard/blob/main/packages/gen-ai/bff/internal/integrations/kubernetes/nemo_guardrails.go)
- [Committed upstream-library PoC](../deploy/phase3/nemo-guardrails-poc.yaml)
- [Fact register](facts.md) and [append-only executed evidence](../EXPERIMENTS.md)
