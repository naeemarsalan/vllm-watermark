# Phase 4 — RHOAI serving and managed NeMo guardrails

**Status: partially complete.** The RHOAI 3.4.2 operator/DSC state, custom runtime
build, ServingRuntime/InferenceService/internal predictor, and managed-NeMo
metadata-only correlation/broker action are `EXECUTED` in scoped forms (facts
C8/D5/D10; [managed-path and outage evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted);
[current build-5 matrix](../../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
The external KServe gateway/Istio pass-through, product supportability, and
platform-wide nonretention remain `OPEN` (facts C8/D6/D10).
Record every rerun command and redacted raw output in the append-only root
[`EXPERIMENTS.md`](../../EXPERIMENTS.md).

## Scope and sources

This directory installs `rhods-operator.3.4.2` from the observed
`stable-3.4` channel, manages only KServe and TrustyAI in the
`DataScienceCluster`, then provides templates for a custom vLLM runtime,
single-model `InferenceService`, and TrustyAI-managed `NemoGuardrails` resource.
The target operator/DSC, runtime, internal predictor, and former managed action
states are recorded as `EXECUTED`; the current `30`/`32` correlation/broker assets
were executed through the managed path with metadata-only broker correlation
(facts C8/D5/D10; [current-path evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
The external gateway test, product supportability, platform-wide nonretention,
and HTTP broker hardening/auth/network policy remain `OPEN`.

- `OFFICIAL-SRC`: [RHOAI 3.4 Operator installation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/installing_and_uninstalling_openshift_ai_self-managed/installing-and-deploying-openshift-ai_install)
- `OFFICIAL-SRC`: [RHOAI 3.4 model-serving configuration](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/pdf/configuring_your_model-serving_platform/Red_Hat_OpenShift_AI_Self-Managed-3.4-Configuring_your_model-serving_platform-en-US.pdf)
- `OFFICIAL-SRC`: [RHOAI 3.4 NeMo Guardrails](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html-single/enabling_ai_safety_with_guardrails/index)
- `STATIC`: the initial RHOAI catalog inspection on 2026-08-08 reported
  `redhat-operators`, `openshift-marketplace`, `stable-3.4`, and
  `rhods-operator.3.4` (`STATIC`; source: [catalog evidence](../../EXPERIMENTS.md#2026-08-08--phase-4-rhoai-evidence-index-reconstructed-not-an-executed-transcript)).
  The recovered run then recorded the operator CSV as `Succeeded`, the DSC as
  `Ready`, and the ServingRuntime/InferenceService/NemoGuardrails APIs as present
  (`EXECUTED`; source: [recovered transcript](../../EXPERIMENTS.md#2026-08-08--phase-4-rhoai-exact-transcript-recovered-executed-redacted)).

## Preconditions

**Basis (`STATIC` safety contract / `EXECUTED` target-cluster state):** the safety
rules come from [`AGENTS.md`](../../AGENTS.md); namespace, detector, Secret-reference,
public-model, and GPU-lifecycle assumptions were checked in the recorded Phase 4/D10
runs ([current evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

- Use a cluster administrator account. RHOAI installation is cluster-scoped.
- Preserve the repository rules in [`AGENTS.md`](../../AGENTS.md): never read,
  print, or commit `aws` or `cluster/`; never log watermark or storage secrets.
- Confirm the existing `watermark` namespace, detector Deployment/Service, and
  `watermark-key` Secret. Both `WATERMARK_KEY` and a non-empty
  `WATERMARK_KEY_ID` are required for generator/detector alignment. Never read
  or print their values. The Phase 4 public `hf://` model URI does not require
  an object-storage data connection.
- Leave GPU replicas at zero through all operator/component installation and
  configuration work. Scale the GPU only immediately before the actual
  `InferenceService` execution run and scale it back to zero afterwards.

## Ordered procedure

The commands below are the reproducible sequence used by the recovered run. The
historical outcome is recorded in `EXPERIMENTS.md`; a rerun must produce fresh
`EXECUTED` evidence and must not inherit prior status (`STATIC`/`EXECUTED`, source:
[recovered transcript](../../EXPERIMENTS.md#2026-08-08--phase-4-rhoai-exact-transcript-recovered-executed-redacted)).

1. Inspect the catalog before applying. It must still show the intended source,
   `stable-3.4`, and `rhods-operator.3.4.2`; otherwise stop and update the
   versioned assets with official-source evidence.

   ```bash
   KUBECONFIG=cluster/auth/kubeconfig oc get packagemanifest rhods-operator \
     -n openshift-marketplace \
     -o jsonpath='{.status.catalogSource} {.status.catalogSourceNamespace}{"\\n"}{range .status.channels[*]}{.name}{" currentCSV="}{.currentCSV}{"\\n"}{end}'
   ```

2. Server-side dry-run and apply the operator assets. `Manual` approval is
   intentional: review the generated InstallPlan before approving it.

   ```bash
   KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server -f deploy/phase4/00-rhods-operator-install.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc apply -f deploy/phase4/00-rhods-operator-install.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc get installplan -n redhat-ods-operator
   KUBECONFIG=cluster/auth/kubeconfig oc get csv -n redhat-ods-operator
   ```

   Approve only the InstallPlan that resolves to `rhods-operator.3.4.2`, then
   wait for the CSV to report `Succeeded`. Do not use an unpinned channel result
   as proof of 3.4.2.

3. Confirm the actual CRD names/schema after the operator installs, then apply
   the minimal component selection and wait for the DSC phase to be `Ready`.

   ```bash
   KUBECONFIG=cluster/auth/kubeconfig oc api-resources | rg 'DataScienceCluster|ServingRuntime|InferenceService|NemoGuardrails'
   KUBECONFIG=cluster/auth/kubeconfig oc explain datasciencecluster.spec.components --api-version=datasciencecluster.opendatahub.io/v2
   KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server -f deploy/phase4/10-datasciencecluster-minimal.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc apply -f deploy/phase4/10-datasciencecluster-minimal.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc get datasciencecluster default-dsc -o yaml
   ```

4. Build the wheel and custom runtime image without embedding any Secret. The
   resulting registry image must be pinned by digest in
   `20-watermark-vllm-servingruntime.yaml` before use. The first recovered build
   pinned base digest
   `sha256:5800e12b2a465f15961fcf34b645d79ed4f91ec9161eab22b1205d12682183c8`
   and produced derived digest
   `sha256:571746d756a6d8671660b98da1f2738616f662822630979e1848b6b1b9ab9683`.
   The current manifest instead pins build `watermark-vllm-3` at
   `sha256:f8294ee0459869e9659b1178ed91f57a1b52a52c6a5f5f819ca651646b317e4c`,
   which is the image used by the executed Phase 4/D10 matrix (`EXECUTED`;
   sources: [recovered first build](../../EXPERIMENTS.md#custom-image-build) and
   [current matrix image identity](../../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
   A rerun must pin its newly observed immutable digest rather than assuming
   either historical digest remains current.

   ```bash
   set -euo pipefail
   ./deploy/phase0/build-wheel.sh
   KUBECONFIG=cluster/auth/kubeconfig oc apply -f deploy/phase4/15-vllm-build.yaml
   mapfile -t phase4_wheels < <(find dist -maxdepth 1 -type f \
     -name 'vllm_watermark-*.whl' -print)
   if [[ ${#phase4_wheels[@]} -ne 1 ]]; then
     printf 'Expected exactly one wheel; found %s\n' "${#phase4_wheels[@]}" >&2
     exit 1
   fi
   phase4_build_context=$(mktemp -d /tmp/vllm-watermark-runtime-build.XXXXXX)
   phase4_wheel_name=$(basename "${phase4_wheels[0]}")
   cleanup_phase4_build_context() {
     set +e
     case "$phase4_build_context" in
       /tmp/vllm-watermark-runtime-build.*)
         unlink "$phase4_build_context/Containerfile" 2>/dev/null
         unlink "$phase4_build_context/dist/$phase4_wheel_name" 2>/dev/null
         rmdir "$phase4_build_context/dist" 2>/dev/null
         rmdir "$phase4_build_context" 2>/dev/null
         ;;
       *)
         printf 'Refusing to remove unexpected path: %s\n' \
           "$phase4_build_context" >&2
         ;;
     esac
   }
   trap cleanup_phase4_build_context EXIT INT TERM
   install -m 0644 deploy/phase4/Containerfile "$phase4_build_context/Containerfile"
   mkdir -p "$phase4_build_context/dist"
   install -m 0644 "${phase4_wheels[0]}" \
     "$phase4_build_context/dist/$phase4_wheel_name"
   find "$phase4_build_context" -type f
   KUBECONFIG=cluster/auth/kubeconfig oc start-build watermark-vllm -n watermark \
     --from-dir="$phase4_build_context" --follow --wait
   KUBECONFIG=cluster/auth/kubeconfig oc get istag watermark-vllm:0.1.0 \
     -n watermark -o jsonpath='{.image.dockerImageReference}{"\n"}'
   cleanup_phase4_build_context
   trap - EXIT INT TERM
   ```

5. Substitute the derived ImageStream `@sha256:...` pull spec in the runtime
   template, and hard-stop if its placeholder remains. The `InferenceService`
   uses the public `hf://Qwen/Qwen2.5-0.5B-Instruct` URI accepted by the live
   RHOAI storage initializer (`EXECUTED`; [recovered Phase 4
   evidence](../../EXPERIMENTS.md#2026-08-08--phase-4-rhoai-exact-transcript-recovered-executed-redacted)).
   Verify both installed CRD schemas before applying.

   ```bash
   KUBECONFIG=cluster/auth/kubeconfig oc explain servingruntime.spec --api-version=serving.kserve.io/v1alpha1
   KUBECONFIG=cluster/auth/kubeconfig oc explain inferenceservice.spec.predictor.model --api-version=serving.kserve.io/v1beta1
   ! rg -q 'replace-with-built-digest|quay.io/example' deploy/phase4/20-watermark-vllm-servingruntime.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server -f deploy/phase4/20-watermark-vllm-servingruntime.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server -f deploy/phase4/21-watermark-vllm-inferenceservice.yaml
   ```

6. Apply the deterministic NeMo ServiceAccount/RBAC name first, then create its
   bounded token Secret without printing the token or Secret YAML. Replace the
   ConfigMap exactly rather than merging it: a prior Phase 3 ConfigMap can have
   stale keys that `oc apply` would preserve. Apply the managed CR only after
   every prerequisite passes server-side validation. The `nemo-watermark`
   resource accepts only the bounded request-correlation metadata emitted by the
   gateway for `kgw` or `synthid`; the executed mixed-scheme result is established
   by the current-path matrix, not by manifest inspection (`STATIC` contract /
   `EXECUTED` outcome; [evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

   ```bash
   set -euo pipefail
   KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server \
     -f deploy/phase4/31-nemo-watermark-rbac.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc apply \
     -f deploy/phase4/31-nemo-watermark-rbac.yaml
   nemo_token_file=$(mktemp /tmp/nemo-watermark-token.XXXXXX)
   cleanup_nemo_token() {
     case "$nemo_token_file" in
       /tmp/nemo-watermark-token.*) unlink "$nemo_token_file" 2>/dev/null || true ;;
       *) printf 'Refusing to remove unexpected path: %s\n' "$nemo_token_file" >&2 ;;
     esac
   }
   trap cleanup_nemo_token EXIT INT TERM
   KUBECONFIG=cluster/auth/kubeconfig oc create token \
     nemo-watermark-serviceaccount --duration=336h > "$nemo_token_file"
   KUBECONFIG=cluster/auth/kubeconfig oc create secret generic \
     nemo-watermark-model-token --from-file=token="$nemo_token_file" \
     -n watermark --dry-run=client -o yaml \
     | KUBECONFIG=cluster/auth/kubeconfig oc apply -f -
   cleanup_nemo_token
   trap - EXIT INT TERM
   KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server -f deploy/phase4/30-nemo-watermark-config.yaml
   if KUBECONFIG=cluster/auth/kubeconfig oc get configmap nemo-watermark-config \
     -n watermark >/dev/null 2>&1; then
     KUBECONFIG=cluster/auth/kubeconfig oc replace \
       -f deploy/phase4/30-nemo-watermark-config.yaml
   else
     KUBECONFIG=cluster/auth/kubeconfig oc create \
       -f deploy/phase4/30-nemo-watermark-config.yaml
   fi
   KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server \
     -f deploy/phase4/32-nemo-watermark.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc apply \
     -f deploy/phase4/32-nemo-watermark.yaml
   KUBECONFIG=cluster/auth/kubeconfig oc get nemoguardrails nemo-watermark -n watermark -w
   ```

   Before running this step in a shared shell, replace the temporary-file
   handling with the approved secret-management method if `/tmp` is not an
   acceptable local transient location. This repo does not assert a retention
   posture for that shell or platform (`OPEN`, D5). The current finite marker scan
   found zero matches in the sampled managed-NeMo, detector, event, and action-log
   surfaces, but this does not establish platform-wide non-retention (`EXECUTED`
   scoped / `OPEN`; source: [current-path evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

7. Only after the platform resources are ready, scale the GPU for the bounded
   execution window, run the Phase 4 positive/clean and pass-through evidence,
   then scale it down in a `finally`-equivalent cleanup path. The actual
   endpoint and internal predictor response shape, managed-NeMo readiness,
   metadata-only broker correlation, detector invocation, latency, and
   closed-policy behavior are recorded as
   `EXECUTED`; external gateway/Istio pass-through remains `OPEN` (facts
   C8/D5/D10; source: [current-path evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

   ```bash
   ./scripts/scale-gpu.sh 1
   # Execute the pre-registered Phase 4 evidence matrix; redact content and secrets.
   ./scripts/scale-gpu.sh 0
   ```

## Explicit boundaries

- The `NemoGuardrails` route is not assumed to be a transparent vLLM proxy.
  RHOAI 3.4 documentation explicitly warns that `/v1/chat/completions` can
  modify or drop request/model-response fields. Therefore it cannot establish
  `vllm_xargs` metadata pass-through by inspection; the recovered direct internal
  predictor test executed, but the external gateway/Istio pass-through remains
  `OPEN` (C8; source: [current-path evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
- The managed NeMo custom action sends metadata only to the broker; the exact
  pending gateway response content, identified by its correlated digest, remains
  the detector authority. The current `30`/`32` files and this route executed
  real KGW/SynthID positive and clean controls, correlated IDs/digests/scheme/key,
  and failed closed after three attempts (two retries) when detector endpoints
  were unavailable, then recovered (D5/D10; source: [current-path evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
  This does not prove that managed NeMo natively implements the legacy
  FMS/TrustyAI detector contract (`OPEN`; facts C11/D5 and the
  [current-path evidence](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
- The recovered 2026-08-08 fixed action was every-response and did not itself
  implement sampling. The later gateway/current action executed the configurable
  selector, queue/retry/metrics behavior, and exact `N=1`/`N=5` matrix
  (`EXECUTED`, scoped; D10 and the [current build-5 matrix](../../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
  Multi-replica/global ordinals and streaming/asynchronous delivery remain `OPEN`.
- Supportability of the custom RHOAI runtime image remains an external D6
  product/support decision even if the deployment executes successfully
  (`OPEN`; facts C4/D6).
