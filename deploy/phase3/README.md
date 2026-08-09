# Phase 3 — detector service, standalone FMS, and upstream NeMo PoC

Manifests in this directory stand up (1) the watermark detector service as a
built-on-cluster image (`detector-build.yaml`, `detector-deploy.yaml`) and
(2) a **standalone** (not RHOAI-managed) FMS Guardrails Orchestrator wired
to it in detection-only mode (`orchestrator.yaml`), on plain OpenShift 4.20
(`ocp-ai`) with **no RHOAI installed**. This is the executed legacy FMS
path, not a recommended future RHOAI architecture (`EXECUTED` /
`OFFICIAL-SRC`; [facts C11/D5](../../docs/facts.md)). A separate committed
configuration exercised an upstream `nemoguardrails==0.23.0` custom action
(`EXECUTED`; [NeMo hardening transcript in the append-only evidence log](../../EXPERIMENTS.md)).
The historical Phase 3 run did not cover the RHOAI-managed `NemoGuardrails` CR,
shipped version, or managed retention behavior. Those were exercised later in
the current internal metadata-only broker path (`EXECUTED`; [current Phase
4/D10 evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
External KServe/Istio pass-through, supportability, and platform-wide retention
remain `OPEN` (same evidence; facts C8/D6/D10).

The FMS build, deployment, health checks, and verdict matrix were executed
on 2026-08-08. Preserved raw outcomes and corrections are in
[`EXPERIMENTS.md`](../../EXPERIMENTS.md); this runbook is the reusable command
path and is not evidence by itself (`EXECUTED`; [Phase 3 evidence](../../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half)).

CPU-only throughout (tokenizer + torch CPU scoring for the detector; a
stateless Rust HTTP proxy for the orchestrator) — **no GPU node needed** in
the recorded Phase 3 configuration (`EXECUTED`; run log), unlike generation
workloads in `deploy/phase0`. Do not run `scripts/scale-gpu.sh 1`
for this phase.

## Prerequisites

```bash
export KUBECONFIG=cluster/auth/kubeconfig   # repo-root relative; gitignored, never print its contents
```

- **Namespace**: this phase reuses the `watermark` namespace created by
  `deploy/phase0/namespace.yaml` (`oc apply -f deploy/phase0/namespace.yaml`
  if starting from a clean cluster — idempotent either way, see that file's
  own comment).
- **Watermark key Secret**: this phase reuses the EXISTING `watermark-key`
  Secret from `deploy/phase0/secret-template.yaml` → `secret.yaml`
  (gitignored). If Phases 0–2 already ran on this cluster, it may already be
  there; if not, follow `deploy/phase0/README.md` §1 to create it before
  continuing (`detector-deploy.yaml`'s env references it by name and will
  sit `CreateContainerConfigError` without it).
- **Detector signing Secret**: both detector Deployments require
  `detector-signing-key` (`STATIC`; manifests). Create it before deploying
  §4 only if it does not already exist, using a key file outside the
  repository. This example reuses an existing Secret and otherwise creates
  one without printing the private key:

  ```bash
  if oc -n watermark get secret detector-signing-key >/dev/null 2>&1; then
    echo 'detector-signing-key already exists; reusing it'
  else
    (
      set -euo pipefail
      umask 077
      signing_key_path=$(mktemp /tmp/vllm-watermark-signing.XXXXXX.pem)
      trap 'rm -f -- "$signing_key_path"' EXIT
      trap 'exit 130' HUP INT TERM
      read -r -p 'Signing key ID: ' signing_key_id
      test -n "$signing_key_id"
      openssl genpkey -algorithm ed25519 -out "$signing_key_path"
      oc -n watermark create secret generic detector-signing-key \
        --from-file=signing.pem="$signing_key_path" \
        --from-literal="SIGNING_KEY_ID=$signing_key_id" \
        --dry-run=client -o yaml | oc apply -f -
    )
  fi
  ```

  Replacing that Secret changes the signing key and identifier consumed by
  both Deployments (`STATIC`; manifests); rotation policy remains `OPEN`
  under D4.
  Key material must never be committed or logged
  ([repository rules](../../AGENTS.md#3-secrets-and-safety)).
- **`pyyaml`**: used below only to validate this directory's own YAML syntax
  locally, not by anything that runs in-cluster. Check availability with
  `python3 -c "import yaml; print(yaml.__version__)"`; no version claim is
  made by this runbook.

## 1. Build the wheel

```bash
./deploy/phase0/build-wheel.sh
# -> dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl (re-run whenever
#    src/vllm_watermark/ changes; see that script's own header comment)
```

## 2. Build context safety — read before running `oc start-build`

`detector/Dockerfile` needs three files under `detector/` plus the single built
wheel under `dist/`. Never submit the repository root: it also contains
gitignored live-credential locations that must not enter a build context
([repository rules](../../AGENTS.md#3-secrets-and-safety)). The current executed
build used the exact four-file allow-list below (`EXECUTED`; [current detector
reconciliation](../../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09)):

```bash
prepare_detector_build_context() {
  local candidate
  local -a detector_wheels
  candidate=$(mktemp -d /tmp/vllm-watermark-detector-build.XXXXXX) || return 1
  if [[ ! "$candidate" =~ ^/tmp/vllm-watermark-detector-build\.[[:alnum:]]{6}$ ]] ||
     [[ ! -d "$candidate" ]]; then
    echo 'mktemp returned an unexpected build-context path; refusing to continue' >&2
    return 1
  fi
  mapfile -t detector_wheels < <(find dist -maxdepth 1 -type f \
    -name 'vllm_watermark-*.whl' -print)
  if [[ ${#detector_wheels[@]} -ne 1 ]] ||
     ! mkdir -p "$candidate/detector" "$candidate/dist" ||
     ! cp -- detector/Dockerfile detector/app.py detector/requirements.txt \
       "$candidate/detector/" ||
     ! cp -- "${detector_wheels[0]}" "$candidate/dist/"; then
    rm -rf -- "$candidate"
    return 1
  fi
  VLLM_WATERMARK_DETECTOR_BUILD_CONTEXT=$candidate
  export VLLM_WATERMARK_DETECTOR_BUILD_CONTEXT
}

if prepare_detector_build_context; then
  # Sanity check: this must print exactly the four allow-listed files above.
  find "$VLLM_WATERMARK_DETECTOR_BUILD_CONTEXT" -type f
else
  unset VLLM_WATERMARK_DETECTOR_BUILD_CONTEXT
  echo 'build-context preparation failed; do not run oc start-build' >&2
fi
unset -f prepare_detector_build_context
```

The repo-root `.dockerignore` is defense in depth for other build tools; this
binary-build procedure relies on the staged allow list, not on assumed ignore
behavior (`STATIC`; `.dockerignore` and commands above).

## 3. Create the ImageStream + BuildConfig, then build

```bash
oc apply -f deploy/phase3/detector-build.yaml
oc -n watermark start-build detector \
  --from-dir="$VLLM_WATERMARK_DETECTOR_BUILD_CONTEXT" --follow
```

Watch for `Push successful` at the end of the follow output, then confirm
the ImageStreamTag exists:

```bash
oc -n watermark get istag detector:latest
```

## 4. Deploy the detector service

```bash
oc apply -f deploy/phase3/detector-deploy.yaml
oc apply -f deploy/phase3/detector-synthid-deploy.yaml   # synthid-scheme twin; see that file's header
oc -n watermark wait --for=condition=Available deploy/detector-synthid --timeout=8m
oc -n watermark wait --for=condition=Available deploy/detector --timeout=5m
```

The Deployment's `image.openshift.io/triggers` annotation (see that file's
header comment for the exact OpenShift 4.20 doc citations) means the
placeholder `image:` field in the manifest gets overwritten with the real
built image automatically once step 3's build completes and updates the
`detector:latest` ImageStreamTag — `oc apply`-ing `detector-deploy.yaml`
*before* step 3 has ever produced a build is fine (the pod sits
`ImagePullBackOff` on the placeholder pull spec until the trigger fires),
but doing step 3 first, as ordered above, avoids that transient state.

Sanity-check that the detector loaded its tokenizer and key configuration:

```bash
oc -n watermark port-forward svc/detector 8000:8000 &
curl -s http://localhost:8000/health
# -> {"status":"ok"}
curl -s http://localhost:8000/ready
# -> {"status":"ready","tokenizer_loaded":true,"key_ids":["<your key_id>"]}
#    (503 with tokenizer_loaded/keys_configured booleans if not yet ready —
#    see detector-deploy.yaml's readinessProbe comment for expected timing)
```

The `/ready` handler checks tokenizer/default-key availability, while lifespan
validates detector numeric configuration before the route can be served
(`STATIC`; `detector/app.py`). The B23 probe showed an earlier revision
accepting NaN/out-of-domain values, and a later review found missing upper
bounds and then an explicit-blank vocabulary bypass after the first remediation.
The current immutable image matched local source, rejected both blank forms,
passed all nine built-image maximum/overflow pairs, failed a controlled
blank-valued rollout before readiness, recovered, and answered both scheme API
smoke requests (`EXECUTED`; [current detector reconciliation](../../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09)).
The full generated-response D10 matrix subsequently reran through this detector
digest (`EXECUTED`; [current build-5 D10 rerun](../../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).

## 5. Deploy the orchestrator

```bash
oc apply -f deploy/phase3/orchestrator.yaml
oc -n watermark wait --for=condition=Available deploy/orchestrator --timeout=3m
```

```bash
oc -n watermark port-forward svc/orchestrator 8034:8034 &
curl -s http://localhost:8034/health
# -> {"fms-guardrails-orchestr8":"0.16.0"}
```

If this 503s or connection-refuses, check `oc -n watermark logs
deploy/orchestrator` first — a config parse error (e.g. a YAML typo in the
`orchestrator-config` ConfigMap) crash-loops the container immediately
rather than starting degraded, since the whole config is loaded once at
process start (`OrchestratorConfig::load`, `src/main.rs`).

## 6. Exercise both endpoints

Scheme routing is SERVER-SIDE: `watermark-kgw` routes to the kgw-scheme
`detector` Deployment and `watermark-synthid` to the dedicated
`detector-synthid` Deployment (each pins `WATERMARK_DETECTOR_SCHEME`), so
**both detector ids return correct verdicts with empty `detector_params`**
(`EXECUTED`; [raw matrix](../../EXPERIMENTS.md#raw-evidence-phase-3-verdict-matrix-signing-retention-health-fresh-re-run)).
The Phase 3 narrative reports that deployed 0.16.0 forwarded a
`detector_params.scheme` override, but its command and raw response were not
preserved, so that specific probe remains `OPEN`
([Phase 3 narrative](../../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half)).
The 0.18.3 source forwards parameters other than `threshold` (`STATIC`;
[API note](../../docs/api-notes-trustyai-detectors.md)). Acceptance uses empty
parameters, so the twin-Deployment design does not depend on either
client-controlled behavior; see `orchestrator.yaml`'s header.

### (a) Direct detector endpoint — `/v1/watermark/detect`

Port-forward from step 4 (`svc/detector` on `localhost:8000`) still needs to
be running.

```bash
# A text with genuinely high z-score is needed to see a `true` verdict —
# substitute a real Phase 1/2 watermarked sample from EXPERIMENTS.md /
# benchmarks/data/ output (gitignored, not checked in) rather than this
# placeholder, which is far too short to score meaningfully (see
# detector/app.py's `InsufficientTokensError` — KGW needs >= 2 tokens,
# SynthID needs >= ngram_len (default 5), or it 422s).
curl -s http://localhost:8000/v1/watermark/detect \
  -H 'Content-Type: application/json' \
  -d '{"text": "<paste a known-watermarked KGW sample here>", "scheme": "kgw"}'
```

Both committed Deployments read `SIGNING_KEY_PATH` and `SIGNING_KEY_ID` from
the prerequisite signing Secret (`STATIC`; manifests). The following fields
and values come from the preserved single-instance KGW direct response
(`EXECUTED`; [raw capture](../../EXPERIMENTS.md#raw-evidence-phase-3-verdict-matrix-signing-retention-health-fresh-re-run)); the detached JWS is elided:

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

### (b) Orchestrator — `POST /api/v2/text/detection/content`

Port-forward `svc/orchestrator` on its main port too:
`oc -n watermark port-forward svc/orchestrator 8033:8033 &`

**KGW** (empty `detector_params`; the dedicated Deployment pins the KGW
scheme):

```bash
curl -s http://localhost:8033/api/v2/text/detection/content \
  -H 'Content-Type: application/json' \
  -d '{
        "content": "<paste a known-watermarked KGW sample here>",
        "detectors": {"watermark-kgw": {}}
      }'
```

**SynthID** (empty params; the dedicated Service makes scheme authority
server-side):

```bash
curl -s http://localhost:8033/api/v2/text/detection/content \
  -H 'Content-Type: application/json' \
  -d '{
        "content": "<paste a known-watermarked SynthID sample here>",
        "detectors": {"watermark-synthid": {}}
      }'
```

The deployed 0.16.0 orchestrator returned this response shape
(`EXECUTED`; same raw capture). Content is elided because the contract echoes
detected text:

```json
{
  "detections": [
    {
      "start": 0,
      "end": 1113,
      "text": "<submitted content elided>",
      "detection": "kgw-watermark",
      "detection_type": "watermark",
      "detector_id": "watermark-kgw",
      "score": 1.0,
      "metadata": {
        "z_score": 6.400354600105544,
        "p_value": 7.750825489048927e-11,
        "key_id": "poc-2026-08",
        "scheme": "kgw",
        "num_tokens_scored": 188,
        "detector_version": "vllm-watermark-detector/0.1.0.dev0",
        "num_green": 85,
        "gamma": 0.25
      }
    }
  ]
}
```

Below the detector's own z-threshold (default z≥4.0), the detector service
returns an empty per-content list (deliberate "no detection" convention —
see `detector/app.py`'s docstring citation of the upstream
`detectors/huggingface/detector.py` behavior it mirrors), so the orchestrator
response for unwatermarked/human text is `{"detections": []}`, not an error.

`ContentAnalysisResponse.detector_id` disambiguation was also `EXECUTED`: one
request naming both detectors returned only the matching detection with the
correct detector id ([raw matrix](../../EXPERIMENTS.md#raw-evidence-phase-3-verdict-matrix-signing-retention-health-fresh-re-run)).

## 7. Upstream NeMo 0.23.0 PoC

[`nemo-guardrails-poc.yaml`](nemo-guardrails-poc.yaml) contains the reusable
upstream-library output rail, flow, and custom action; the resolved package
set is pinned in [`nemo-poc-constraints.txt`](nemo-poc-constraints.txt)
(`STATIC`; committed artifacts). It is not a RHOAI `NemoGuardrails` CR.

The recorded fresh-pod run installed the 78 pinned packages with zero freeze
differences, blocked a known KGW sample, passed a human sample through
`POST /v1/checks`, blocked missing/non-boolean verdict responses, and kept
five poisoned detector fields out of the captured process output
(`EXECUTED`; [NeMo hardening transcript in the append-only evidence log](../../EXPERIMENTS.md)).
The committed detector-outage branch is `STATIC`; this historical upstream-PoC
run did not preserve a live outage command/raw output. The current managed
path's real detector outage and fail-closed recovery were executed separately
(`EXECUTED`; [current Phase 4/D10 evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

This PoC does not close RHOAI operator mounting, shipped-version, retention,
or supportability questions (`OPEN`; C11/D5/D6). The upstream library's 422
and event-log content-handling gaps also require mitigation before reuse; no
zero-retention claim is made ([NeMo API note](../../docs/api-notes-nemo-guardrails.md)).

## 8. Teardown

```bash
oc -n watermark delete -f deploy/phase3/orchestrator.yaml --ignore-not-found
oc -n watermark delete -f deploy/phase3/detector-synthid-deploy.yaml --ignore-not-found
oc -n watermark delete -f deploy/phase3/detector-deploy.yaml --ignore-not-found
oc -n watermark delete -f deploy/phase3/detector-build.yaml --ignore-not-found
oc -n watermark delete -f deploy/phase3/nemo-guardrails-poc.yaml --ignore-not-found
build_context=${VLLM_WATERMARK_DETECTOR_BUILD_CONTEXT:-}
if [[ -z "$build_context" ]]; then
  echo 'build-context variable is unset; no local directory removed'
elif [[ "$build_context" =~ ^/tmp/vllm-watermark-detector-build\.[[:alnum:]]{6}$ ]] &&
     [[ "$(dirname -- "$build_context")" == /tmp ]] &&
     [[ -d "$build_context" ]]; then
  rm -rf -- "$build_context"
  unset VLLM_WATERMARK_DETECTOR_BUILD_CONTEXT
else
  echo 'refusing to remove an unexpected or missing build-context path' >&2
fi
unset build_context
```

These commands leave the `watermark` namespace, `watermark-key`, and
`detector-signing-key` in place. Remove those Secrets explicitly when they
are no longer required; never print their contents. No GPU node was used by
the recorded Phase 3 work (`EXECUTED`; run log), so this runbook does not
scale one down.

## Acceptance status

The Phase 3 executable PoC scope is met: the standalone legacy FMS path and
the upstream NeMo 0.23.0 path returned the recorded expected verdicts
(`EXECUTED`; fact D5 and [`EXPERIMENTS.md`](../../EXPERIMENTS.md)). The later
RHOAI-managed path/initial D10 matrix, subsequent D9 startup-validation rebuild,
and current-image matrix rerun are recorded separately as scoped `EXECUTED`
evidence ([current Phase 4/D10
evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted);
[current detector reconciliation](../../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09);
[current build-5 D10 rerun](../../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
External gateway/Istio pass-through, supportability, key lifecycle, and
production hardening remain `OPEN` (same evidence; facts C4/C8/D4/D6/D10). This
runbook itself does not establish RHOAI completion or production readiness.

## YAML validation

```bash
python3 -c "
import pathlib, yaml
for p in sorted(pathlib.Path('deploy/phase3').glob('*.yaml')):
    docs = list(yaml.safe_load_all(p.read_text()))
    print(f'{p}: {len(docs)} document(s) OK')
"
```

Re-run the command above for the current result; it parses all five YAML
manifests. The repo-root `.dockerignore` is plain text and is not part of this
check. This confirms only that the files parse as valid YAML. The stronger
`oc apply --dry-run=server` check against the live OpenShift 4.20 API was
also executed for every Phase 3 object; its raw output and the bare vLLM
PodSecurity warning are recorded in `EXPERIMENTS.md`.
