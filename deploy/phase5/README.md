# Phase 5 D10 validation gateway

**Status (`EXECUTED`, scoped; source: [2026-08-09 execution record](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)):** The build, deployment, and current D10
continuous-validation run executed on 2026-08-09. Exact commands and
content-redacted raw output are recorded in the append-only root
[`EXPERIMENTS.md`](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted).
This evidence does not close the explicitly listed `OPEN` boundaries below.

## Source-reviewed contract

**Claim (`STATIC`; source:
[`validation/main.py`](../../validation/main.py),
[`validation/http_clients.py`](../../validation/http_clients.py), and
[`validation/gateway.py`](../../validation/gateway.py)):** the deployable
entrypoint is `python -m validation.main`. It fails during construction when a
required setting is absent or malformed, requires TLS verification, and appends
the exact OpenAI and managed-NeMo endpoint paths to configured base URLs. The
current gateway is synchronous and non-streaming, accepts one choice, and uses
the broker detector result rather than inferring a watermark verdict from
NeMo's outer action state.

**Claim (`STATIC`; source: [`validation/gateway.py`](../../validation/gateway.py)):**
the HTTP surface is:

- `POST /v1/completions` and `POST /v1/chat/completions`;
- `GET /health` for process liveness, `GET /ready` for initialized-service
  readiness, and `GET /metrics`;
- bearer-authenticated `POST /internal/v1/guardrail-action` using the broker
  token; and
- bearer-authenticated reset, status, records, configuration validation,
  fault, consumer, and redacted-event routes below
  `/v1/continuous-validation/`, using a separate admin token.

**Claim (`STATIC`; source: [`00-validation-gateway.yaml`](00-validation-gateway.yaml)
and [`validation/gateway.py`](../../validation/gateway.py)):** the base manifest
sets the positive-result delivery policy to `flag`, validation failures to
fail closed, and test controls to `off`. The normal proxy endpoints themselves
do not currently enforce application-level authentication. This template
therefore exposes only a cluster-internal `ClusterIP` Service and adds neither
a Route nor a NetworkPolicy. The intended caller boundary and an externally
reachable authentication and authorization layer remain `OPEN` before any
Route is added. The managed action can report `blocked` while the gateway's
positive delivery policy is `flag`: the former is the managed-NeMo action
result, while the latter controls whether the generated response is delivered.

## Exact runtime configuration

**Claim (`STATIC`; source: `GatewayConfig.from_environment` and
`RuntimeConfig.from_environment` in the source files above):** the manifest
supplies every currently required field. Secret values are referenced, never
embedded.

| Setting | Manifest source | Contract |
|---|---|---|
| `VALIDATION_UPSTREAM_URL` | literal internal Service base | `http://watermark-vllm-predictor.watermark.svc.cluster.local:8080`; the client appends `/v1/completions` or `/v1/chat/completions` |
| `VALIDATION_UPSTREAM_TOKEN` | absent by default | optional internal bearer credential; when absent the client omits `Authorization` |
| `VALIDATION_DETECTOR_URL` | literal full internal endpoint | `http://detector.watermark.svc.cluster.local:8000/v1/watermark/detect` |
| `VALIDATION_DETECTOR_TOKEN` | absent by default | optional internal bearer credential; when absent the client omits `Authorization` |
| `VALIDATION_NEMO_URL` | literal internal Service base | `https://nemo-watermark.watermark.svc`; the client appends `/v1/guardrail/checks` |
| `VALIDATION_NEMO_TOKEN` | `watermark-validation-gateway-auth/nemo-token` | required managed-NeMo bearer credential |
| `VALIDATION_NEMO_CONFIG_ID` | literal | `watermark-validation`, matching the Phase 4 ConfigMap/CR name |
| `VALIDATION_NEMO_MODEL` | literal | `watermark-vllm`, matching the served-model name |
| `VALIDATION_BROKER_TOKEN` | `watermark-validation-gateway-auth/broker-token` | authenticates only the managed custom-action broker call |
| `VALIDATION_ADMIN_TOKEN` | `watermark-validation-gateway-auth/admin-token` | separate credential for all continuous-validation administration routes |
| `WATERMARK_KEY_ID` | `watermark-key/WATERMARK_KEY_ID` | non-secret identifier only; the gateway never mounts watermark key material |

**Claim (`STATIC` configuration / `EXECUTED` managed-TLS call; source:
[`gateway-entrypoint.sh`](gateway-entrypoint.sh),
[`00-validation-gateway.yaml`](00-validation-gateway.yaml), `HttpClientSettings`,
and the [execution record](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)):**
`VALIDATION_TLS_VERIFY=true` cannot be disabled. The
`watermark-validation-gateway-service-ca` ConfigMap is annotated for OpenShift
Service CA injection, and the manifest always points `VALIDATION_TLS_CA_BUNDLE`
at its projected `service-ca.crt`. The runtime validates that bundle during
startup, so an absent or malformed injected bundle prevents readiness rather
than silently falling back for the internal managed-NeMo TLS call. Whether the
installed managed-NeMo Service certificate validates against this injected CA
was executed successfully; the service CA was mounted and managed TLS calls
completed with verification enabled.

**Authentication contract (`STATIC`; source:
[`00-validation-gateway.yaml`](00-validation-gateway.yaml),
[`validation/gateway.py`](../../validation/gateway.py), and
[`validation/http_clients.py`](../../validation/http_clients.py)):** create or
rotate the referenced authentication Secret through the approved secret
workflow. Do not put their values in a manifest, command-line argument, shell
history, log, or `EXPERIMENTS.md`. The `watermark-validation-gateway-auth`
Secret must contain exactly the three keys named in the table (`nemo-token`,
`broker-token`, and `admin-token`). There is no upstream or detector Secret
key: those internal ClusterIP calls omit `Authorization` unless an operator
explicitly configures a non-empty token. The Phase 4 managed action must use this
exact broker URL and the same `broker-token` key:

```text
http://watermark-validation-gateway.watermark.svc.cluster.local:8080/internal/v1/guardrail-action
```

The gateway calls managed NeMo with the configured `nemo-token`; Kubernetes
RBAC does not authorize HTTP bearer calls, so the gateway ServiceAccount has no
`view` RoleBinding and its API token remains disabled. Add a narrowly scoped
binding only if a future implementation reads Kubernetes resources directly.

## Build an immutable deployment image

**Claim (`STATIC`; source: [`Containerfile`](Containerfile),
[`10-validation-gateway-build.yaml`](10-validation-gateway-build.yaml), and
[`validation/requirements.txt`](../../validation/requirements.txt)):** the
binary BuildConfig installs the pinned top-level FastAPI/httpx/uvicorn runtime
requirements and copies only the four validation runtime modules plus the
entrypoint. The curated context excludes tests and the repository's gitignored
credential paths. The linux/amd64 `python:3.12-slim` base resolved to and is
pinned as
`sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36`
(`EXECUTED`; [direct official-registry inspection](../../EXPERIMENTS.md#2026-08-08--phase-5-gateway-base-image-digest-resolution-executed)):

```text
$ skopeo inspect --override-os linux --override-arch amd64 --format '{{.Digest}} {{.Name}} {{.Architecture}} {{.Os}}' docker://docker.io/library/python:3.12-slim
sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 docker.io/library/python amd64 linux
$ skopeo inspect --override-os linux --override-arch amd64 --format '{{.Digest}} {{.Architecture}} {{.Os}}' docker://docker.io/library/python@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36
sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 amd64 linux
```

**Build result (`EXECUTED`; source: [2026-08-09 execution
record](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)):**
Build5 completed on 2026-08-09 and produced the immutable gateway digest pinned
in the deployment. For a rerun, use the same curated-context `oc start-build
... --from-dir=... --follow --wait` procedure captured in the linked experiment,
then inspect the resulting ImageStream digest and update the deployment only
with that immutable reference. Keep Secret values out of commands, logs, and
experiment output; never submit the repository root to a binary build.

```bash
set -euo pipefail
KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server \
  -f deploy/phase5/10-validation-gateway-build.yaml
KUBECONFIG=cluster/auth/kubeconfig oc apply \
  -f deploy/phase5/10-validation-gateway-build.yaml

phase5_build_context=$(mktemp -d /tmp/vllm-watermark-gateway-build.XXXXXX)
cleanup_phase5_build_context() {
  set +e
  case "$phase5_build_context" in
    /tmp/vllm-watermark-gateway-build.*)
      find "$phase5_build_context" -type f -delete
      find "$phase5_build_context" -depth -type d -empty -delete
      ;;
    *)
      printf 'Refusing to remove unexpected path: %s\n' \
        "$phase5_build_context" >&2
      ;;
  esac
}
trap cleanup_phase5_build_context EXIT INT TERM

install -m 0644 deploy/phase5/Containerfile \
  "$phase5_build_context/Containerfile"
install -m 0755 deploy/phase5/gateway-entrypoint.sh \
  "$phase5_build_context/gateway-entrypoint.sh"
install -m 0644 validation/requirements.txt \
  "$phase5_build_context/requirements.txt"
mkdir -p "$phase5_build_context/validation"
for phase5_source in __init__.py gateway.py http_clients.py main.py; do
  install -m 0644 "validation/$phase5_source" \
    "$phase5_build_context/validation/$phase5_source"
done
find "$phase5_build_context" -type f -printf '%P\n' | sort

KUBECONFIG=cluster/auth/kubeconfig oc start-build \
  watermark-validation-gateway -n watermark \
  --from-dir="$phase5_build_context" --follow --wait
KUBECONFIG=cluster/auth/kubeconfig oc get istag \
  watermark-validation-gateway:0.1.0 -n watermark \
  -o jsonpath='{.image.dockerImageReference}{"\n"}'

cleanup_phase5_build_context
trap - EXIT INT TERM
```

**Immutable deployment (`EXECUTED`; source: [2026-08-09 execution
record](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)):**
the executed deployment uses trusted registry digest
`sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6`.
A mutable tag must not be substituted on rerun (`STATIC` safety requirement;
source: the digest-pinned [`00-validation-gateway.yaml`](00-validation-gateway.yaml)).

## Deploy and run acceptance

**Claim (`STATIC` design / `EXECUTED` deployment result; source:
[`00-validation-gateway.yaml`](00-validation-gateway.yaml),
[`validation/gateway.py`](../../validation/gateway.py), and the
[execution record](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)):**
SQLite owns one
persistent sampler ordinal, so this initial design explicitly enforces one
replica, `Recreate`, and a ReadWriteOnce PVC. Runtime writes are limited to the
PVC and `/tmp`; the root filesystem is read-only. The Pod spec requests neither
a fixed `runAsUser` nor a fixed `fsGroup`; the image's UID 1001 is only its
non-root default, and application files are readable by an arbitrary admitted
UID. The executed deployment was `1/1` with `Recreate`; its ReadWriteOnce PVC
was `Bound`, arbitrary UID admission succeeded, the root filesystem was
read-only, the SQLite database was mode `600`, and the service CA was mounted.

**Claim (`OFFICIAL-SRC`; source: [Kubernetes disruption
documentation](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/#pod-disruption-budgets)) / operational result `OPEN`:** a PodDisruptionBudget
limits simultaneous voluntary disruptions. This single-replica template does
not include one: it cannot demonstrate disruption availability, and a
`minAvailable: 1` policy would permit zero voluntary disruptions. The desired
maintenance policy must be decided before production use.

For a rerun, validate against the installed APIs, then apply the base with test
controls disabled. Keep the immutable image digest check in your operator
procedure and do not substitute a tag:

```bash
set -euo pipefail
KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server \
  -k deploy/phase5
KUBECONFIG=cluster/auth/kubeconfig oc apply -k deploy/phase5
KUBECONFIG=cluster/auth/kubeconfig oc rollout status \
  deployment/watermark-validation-gateway -n watermark --timeout=180s
```

**Claim (`STATIC` control contract / `EXECUTED` bounded use; source: the
strategic-merge [`acceptance/test-controls.yaml`](acceptance/test-controls.yaml)
patch, [`validation/gateway.py`](../../validation/gateway.py), and the
[execution record](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)):**
fault injection and
consumer pause/resume reject requests while test controls are off. The
acceptance patch changes only `VALIDATION_TEST_CONTROLS` to `on`; apply it
only for the bounded fault/backpressure cases, then immediately re-apply the
base and verify the environment returned to `off` without printing Secret
values.

```bash
KUBECONFIG=cluster/auth/kubeconfig oc patch --dry-run=server \
  deployment/watermark-validation-gateway -n watermark --type=strategic \
  --patch-file deploy/phase5/acceptance/test-controls.yaml
KUBECONFIG=cluster/auth/kubeconfig oc patch \
  deployment/watermark-validation-gateway -n watermark --type=strategic \
  --patch-file deploy/phase5/acceptance/test-controls.yaml
# Rerun the reviewed, content-redacted D10 N=1/N=5 command from the linked
# experiment; never print Secret values or generated content.
KUBECONFIG=cluster/auth/kubeconfig oc apply -k deploy/phase5
KUBECONFIG=cluster/auth/kubeconfig oc rollout status \
  deployment/watermark-validation-gateway -n watermark --timeout=180s
```

**Executed status (`EXECUTED`; sources: [initial managed-path
run](../../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)
and [current detector rerun](../../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)):**
gateway build 5 completed; the managed TLS path, exact runtime-hash comparison,
unsampled baseline (`4`), `N=1` (`20/20`) and `N=5` (`20/100`) runs, faults,
queue overflow, metrics, finite scans, actual detector outage, and cleanup with
controls `off` and GPU scaled to `0` all passed. The fixed matrix subsequently
reran against detector build 5 with the complete 40-row mode-bearing, hash-only projection
retained.

**Remaining status (`OPEN`; source: fact D10 and acceptance boundaries):**
caller authentication, Route and NetworkPolicy, mTLS, multi-replica/global
sampling, PDB/HA, an external KServe gateway, supportability, and platform-wide
retention remain open. A managed `blocked` action is separate from the
gateway positive-delivery `flag`; it does not mean the gateway blocked delivery
under the executed `flag` policy.
