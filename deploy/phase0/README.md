# Phases 0–2 — bare-pod vLLM on the GPU node (pre-RHOAI)

Manifests in this directory stand up plain `vllm serve` (Phase 0 baseline)
and the same thing with the KGW and SynthID watermark plugins wired in
(Phases 1/2), as one-off pods directly on the cluster's GPU node — **not** through
OpenShift AI / RHOAI's ServingRuntime/InferenceService machinery, which is
Phase 4 ([implementation plan](../../docs/implementation.md)). This path was
executed for the Phase 0 baseline, corrected single-instance KGW, and SynthID
on 2026-08-08 (`EXECUTED`; [facts C9, D1, D2, and D8](../../docs/facts.md),
[run log](../../EXPERIMENTS.md)). It is not an RHOAI deployment.

## Prerequisites

```bash
export KUBECONFIG=cluster/auth/kubeconfig   # repo-root relative; gitignored, never print its contents
./scripts/scale-gpu.sh 1                     # billable GPU node — see AGENTS.md #3
```

Wait for the GPU MachineSet's Machine/Node to become Ready
(`oc get nodes -o wide`, `oc -n openshift-machine-api get machineset`)
before proceeding — pods below will sit `Pending` until it is.

## 1. Namespace, Service, Secret

```bash
oc apply -f deploy/phase0/namespace.yaml
oc apply -f deploy/phase0/vllm-service.yaml

if test -e deploy/phase0/secret.yaml; then
  echo 'deploy/phase0/secret.yaml already exists; not overwriting it'
else
  install -m 600 deploy/phase0/secret-template.yaml deploy/phase0/secret.yaml
  echo 'edit the gitignored secret.yaml and replace its placeholder before continuing'
fi
```

Edit `deploy/phase0/secret.yaml` before running the next block. The validation
runs in a subshell, so a placeholder produces a failing status without exiting
the caller's interactive shell (`STATIC`; command structure below):

```bash
(
  set -eu
  if oc -n watermark get secret watermark-key >/dev/null 2>&1; then
    echo 'watermark-key already exists; reusing it'
  elif grep -q 'REPLACE_ME_WITH_HEX_SECRET' deploy/phase0/secret.yaml; then
    echo 'refusing to apply secret.yaml while its placeholder remains' >&2
    exit 1
  else
    oc apply -f deploy/phase0/secret.yaml
  fi
)
```

Applying replacement data to the cluster Secret changes the watermark key
consumed by the pods (`STATIC`; manifests); rotation and corpus/key-version
coordination remain `OPEN` under D4.
Generate key material without putting it in shell history or logs, and never
commit the gitignored file ([repository rules](../../AGENTS.md#3-secrets-and-safety)).

Within this Phases 0–2 serving path, only
[`vllm-watermark-pod.yaml`](vllm-watermark-pod.yaml) (§3) consumes the Secret;
the baseline pod (§2) needs no key. The Phase 3
[`detector`](../phase3/detector-deploy.yaml) and
[`detector-synthid`](../phase3/detector-synthid-deploy.yaml) Deployments are
separate consumers (`STATIC`; linked manifests).

## 2. Phase 0 baseline pod

```bash
oc apply -f deploy/phase0/vllm-baseline-pod.yaml
oc -n watermark wait --for=condition=Ready pod/vllm-baseline --timeout=15m
```

`--timeout=15m` is a conservative runbook timeout, not a cold-start claim.
It includes image pull and model download into the `/models-cache` emptyDir. Consult
the recorded Phase 0 run in `EXPERIMENTS.md` for the observed execution rather
than treating this conservative timeout as a startup-time measurement.

Smoke-test from inside the cluster (see §4 for the `bench` pod) or via a
temporary port-forward:

```bash
oc -n watermark port-forward svc/vllm 8000:8000 &
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

## 3. Phases 1–2 watermark pod — wheel-injection sequence

The watermark pod's container starts and **blocks** waiting for a
sentinel file; it will not begin loading the model until you complete
this sequence. See `vllm-watermark-pod.yaml`'s header comment for why
this shape was chosen over the alternatives considered (ConfigMap of the
whole package, a version-matched initContainer, etc.).

```bash
# 1. Build the py3-none-any wheel locally (`STATIC`; pyproject/build script).
#    The output path is
#    dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl; its size changes with
#    the package. Re-run this whenever src/vllm_watermark/ changes.
./deploy/phase0/build-wheel.sh

# 2. Delete vllm-baseline first if it's still running — the GPU node has
#    exactly one nvidia.com/gpu, so vllm-watermark will sit Pending
#    otherwise (see vllm-service.yaml comment for why this constraint
#    exists and why it's fine).
oc -n watermark delete pod vllm-baseline --ignore-not-found

# 3. Create the watermark pod (it will sit NotReady, printing
#    "[vllm-watermark] waiting for /plugin/ready ..." — confirm with
#    `oc -n watermark logs vllm-watermark` before proceeding).
oc apply -f deploy/phase0/vllm-watermark-pod.yaml
oc -n watermark wait --for=condition=PodScheduled pod/vllm-watermark --timeout=5m

# 4. Copy the wheel in, then flip the sentinel.
oc -n watermark cp dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl \
    vllm-watermark:/plugin/vllm_watermark-0.1.0.dev0-py3-none-any.whl
oc -n watermark exec vllm-watermark -- touch /plugin/ready

# 5. Wait for startup, then inspect the captured startup logs.
oc -n watermark wait --for=condition=Ready pod/vllm-watermark --timeout=15m
oc -n watermark logs vllm-watermark
```

Sanity-check the plugin actually loaded (should appear in the startup
logs from both processors' `logger.info(...)` calls):

```bash
oc -n watermark logs vllm-watermark \
  | grep -E "(KGW|SynthID)LogitsProcessor initialized"
```

The wheel registers KGW and SynthID as entry points, and the manifest
intentionally passes no `--logits-processors` flag. Adding the KGW FQCN as
well as installing the wheel loads KGW twice; the earlier delta≈4 signal and
active-path overhead measurements are superseded (`EXECUTED`; [double-load
correction](../../EXPERIMENTS.md#2026-08-08--correction-phase-1-ran-two-kgw-processor-instances-effective-delta-40)).
Use `vllm_xargs.watermark_scheme` to select `kgw` or `synthid` per request
(`EXECUTED`; [corrected Phase 1/2 run](../../EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8)).

Then run the negative/positive request protocols from
`docs/implementation.md` Phases 1 and 2 (KGW, SynthID, temperature 0,
structured output, and the recorded corpora) through `svc/vllm` — the same
Service and DNS name as §2, since only one of the two pods is up at a time.
Speculative-decoding rejection is a separate startup probe, not a request to
the running Service; reproduce it only as documented in the B7 experiment.

## 4. Benchmark execution (in-cluster, not port-forward)

The recorded performance runs executed `benchmarks/*.py` inside the cluster
against `http://vllm:8000/v1` (`EXECUTED`; [Phase 0 and corrected Phase 1/2
records](../../EXPERIMENTS.md)). Keep that path for comparable reruns. A local
port-forward adds an unmeasured hop, so results collected through it must not
be compared to the recorded in-cluster baseline without qualification.

```bash
oc apply -f deploy/phase0/bench-pod.yaml
oc -n watermark wait --for=condition=Ready pod/bench --timeout=2m

# One-time setup inside the bench pod (python:3.12-slim ships neither
# `requests` nor `pip`'s cache primed — confirmed by reading
# bench_serving.py/gen_corpus.py/fetch_human_corpus.py's own imports:
# stdlib + `requests` only, no torch/transformers needed for these three):
oc -n watermark exec bench -- pip install --no-cache-dir requests

# Copy the benchmark scripts and prompt/data files in (repo-root
# relative local path -> pod:/bench):
oc -n watermark cp benchmarks bench:/bench

# Run, e.g.:
oc -n watermark exec bench -- env OPENAI_BASE_URL=http://vllm:8000/v1 \
    python3 /bench/bench_serving.py \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --prompts-file /bench/prompts.txt \
      --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
      --out /tmp/results_baseline.json

# Pull results back out for EXPERIMENTS.md:
oc -n watermark cp bench:/tmp/results_baseline.json ./results_baseline.json
```

`benchmarks/analyze_detection.py` and `benchmarks/bench_greenlist.py`
import `vllm_watermark` directly (`kgw.core`, `kgw.detector`, `keys`) —
those two do **not** need to run inside the cluster at all: per
AGENTS.md §5 ("Local is fine for detector math, unit tests, docs"), the
recommended flow is `oc cp` the generated JSONL corpora (watermarked +
unwatermarked + human) back to the workstation and run
`analyze_detection.py` there against the local `transformers==4.57.6`
install, rather than installing the wheel a second time inside `bench`.
That local detector/test path is covered by the recorded test and analysis
runs (`EXECUTED` at the recorded revision; facts B21/D1/D8).

## 5. Teardown

```bash
oc -n watermark delete pod vllm-baseline vllm-watermark bench --ignore-not-found
oc delete -f deploy/phase0/vllm-service.yaml --ignore-not-found
# Namespace and Secret left in place deliberately unless you are removing
# the whole spike — delete
# deploy/phase0/secret.yaml's resource manually if so, it's gitignored
# and won't be re-created by `oc apply -f deploy/phase0/`.

./scripts/scale-gpu.sh 0   # REQUIRED before ending the work session — AGENTS.md #3, billable node
```

## SCC (Security Context Constraints)

The Phase 0 and watermark pods reached Ready in the recorded runs after
writable cache locations were configured (`EXECUTED`; [Phase 0
record](../../EXPERIMENTS.md#2026-08-08--phase-0-baseline-serving--benchmark-executed),
[corrected watermark run](../../EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8)).
That does not establish a hardened reusable security context. A later
server-side dry run emitted warnings for missing explicit
`allowPrivilegeEscalation: false`, dropped capabilities, `runAsNonRoot`, and
seccomp settings (`EXECUTED`; [warning transcript](../../EXPERIMENTS.md#2026-08-08--independent-post-push-review-correction)).
The later Phase 4 ServingRuntime manifest explicitly sets the restricted-profile
container fields (`STATIC`; `deploy/phase4/20-watermark-vllm-servingruntime.yaml`),
but that does not retrofit these historical Phase 0 Pod manifests. Their reusable
hardening remains `OPEN`; do not grant a broader SCC as a default workaround
([adversarial finding](../../ADVERSARIAL_REVIEW.md#3-podsecurity-hardening-is-incomplete--high-for-reusable-deployment)).

## Resource-sizing note

The pod requests and limits are defined in the manifests (`STATIC`;
[`vllm-baseline-pod.yaml`](vllm-baseline-pod.yaml) and
[`vllm-watermark-pod.yaml`](vllm-watermark-pod.yaml)). No validated memory-
headroom calculation is registered; capacity and eviction behavior remain
`OPEN` for a reusable RHOAI deployment.

## Image digest evidence

```
$ curl -s https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.18.0
-> HTTP 200; tag_status "active"; digest (manifest list):
   sha256:c32358ebfc115d56ade2acfdbcd00df5b115417dbd6006547c88f07e2b39de06
   images: [amd64 sha256:96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05,
            arm64 sha256:be723f3fa62508d6be28295d86de4aec9791e275474b82ed4697479242948e4d]
```

The registry lookup and deployed manifest-list digest are preserved in the
Phase 0 log (`OFFICIAL-SRC` for the registry response; `EXECUTED` for the
deployment pin; [source](../../EXPERIMENTS.md#2026-08-08--phase-0-infrastructure-bring-up-ocp-ai-cluster)).
