# Phase 0/1 — bare-pod vLLM on the GPU node (pre-RHOAI)

Manifests in this directory stand up plain `vllm serve` (Phase 0 baseline)
and the same thing with the KGW and SynthID watermark plugins wired in
(Phases 1/2), as
one-off pods directly on the cluster's GPU node — **not** through
OpenShift AI / RHOAI's ServingRuntime/InferenceService machinery, which is
Phase 4 (`docs/implementation.md`). This runbook was executed for Phases 0
and 1 on 2026-08-08; commands and raw output are recorded in
`EXPERIMENTS.md`, while the manifests below remain the reproducible source.

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

cp deploy/phase0/secret-template.yaml deploy/phase0/secret.yaml   # gitignored — see deploy/phase0/.gitignore
# edit secret.yaml: replace REPLACE_ME_WITH_HEX_SECRET with a real hex
# key (e.g. `python3 -c "import secrets; print(secrets.token_hex(32))"`
# — run locally, never paste the output into a place that gets logged)
oc apply -f deploy/phase0/secret.yaml
```

The Secret is only consumed by `vllm-watermark-pod.yaml` (§3) — the
baseline pod (§2) needs no key.

## 2. Phase 0 baseline pod

```bash
oc apply -f deploy/phase0/vllm-baseline-pod.yaml
oc -n watermark wait --for=condition=Ready pod/vllm-baseline --timeout=15m
```

`--timeout=15m` is generous, not a claim about actual cold-start time —
first-run image pull is ~9.5GiB (see digest evidence in the pod manifest)
plus model download into the `/models-cache` emptyDir on top of that. Consult
the recorded Phase 0 run in `EXPERIMENTS.md` for the observed execution rather
than treating this conservative timeout as a startup-time measurement.

Smoke-test from inside the cluster (see §4 for the `bench` pod) or via a
temporary port-forward:

```bash
oc -n watermark port-forward svc/vllm 8000:8000 &
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models
```

## 3. Phase 1 watermark pod — wheel-injection sequence

The watermark pod's container starts and **blocks** waiting for a
sentinel file; it will not begin loading the model until you complete
this sequence. See `vllm-watermark-pod.yaml`'s header comment for why
this shape was chosen over the alternatives considered (ConfigMap of the
whole package, a version-matched initContainer, etc.).

```bash
# 1. Build the wheel locally (pure-Python; matches ANY Python 3.11+/3.14
#    interpreter, so this does not need to match the pod's Python
#    version — see build-wheel.sh header comment). The output path is
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

# 5. Watch it install the wheel and start vllm serve.
oc -n watermark logs -f vllm-watermark
oc -n watermark wait --for=condition=Ready pod/vllm-watermark --timeout=15m
```

Sanity-check the plugin actually loaded (should appear in the startup
logs from `KGWLogitsProcessor.__init__`'s `logger.info(...)` call — see
`src/vllm_watermark/kgw/processor.py`):

```bash
oc -n watermark logs vllm-watermark | grep -i "KGWLogitsProcessor initialized"
```

Then run the negative/positive test protocol from
`docs/implementation.md` Phase 1 (temperature 0, structured-output
request, spec-decode flag, ≥100 watermarked/unwatermarked/human-corpus
generations) through `svc/vllm` — same Service, same DNS name, as §2,
since only one of the two pods is ever up at a time (see
`vllm-service.yaml`).

## 4. Benchmark execution (in-cluster, not port-forward)

Recommendation: run `benchmarks/*.py` from inside the cluster against
`http://vllm:8000/v1` (the Service's in-cluster DNS name), not through a
local `oc port-forward`. Reasons: (a) `bench_serving.py`'s own docstring
says it's "designed to run inside a lightweight bench pod" — this was an
existing design decision in the benchmark script, not invented here; (b)
throughput/latency numbers measured through a port-forward include the
port-forward tunnel's own overhead and are not representative of
production request paths; (c) a long benchmark run (≥100 generations ×
256 tokens, per the Phase 1 acceptance criteria) is a poor fit for a
foreground `port-forward` process that dies if your local
network/terminal session hiccups.

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
(It's a plain `pip install .` locally if you do want it in-cluster too —
not covered further here since it wasn't needed for anything in this
task.)

## 5. Teardown

```bash
oc -n watermark delete pod vllm-baseline vllm-watermark bench --ignore-not-found
oc delete -f deploy/phase0/vllm-service.yaml --ignore-not-found
# Namespace and Secret left in place deliberately (cheap, no billing
# impact) unless you're tearing down the whole spike — delete
# deploy/phase0/secret.yaml's resource manually if so, it's gitignored
# and won't be re-created by `oc apply -f deploy/phase0/`.

./scripts/scale-gpu.sh 0   # REQUIRED before ending the work session — AGENTS.md #3, billable node
```

## SCC (Security Context Constraints)

No SCC changes are anticipated. `oc get scc` confirms `restricted-v2`
exists cluster-wide (the OpenShift 4.20 default); none of the manifests
here request `privileged`, `hostPath`, `hostNetwork`, extra Linux
capabilities, or a specific `runAsUser`/`fsGroup` — all deliberately
left for OpenShift's default arbitrary-UID assignment rather than
hardcoded.

**One open, unverified risk** (flagged honestly rather than guessed
around, since this task could not start a pod to check): `skopeo
inspect` shows the `vllm-openai` image sets no `USER` (so it runs as
root, UID 0, inside the container — this is normal for upstream
non-Red-Hat images, not something specific to this pin). Under
`restricted-v2`, OpenShift will still assign an arbitrary non-root UID
at runtime regardless of what the image's default USER is — that part
is standard and fine. What's *not* independently confirmed here is
whether every directory the container needs to write to at runtime
(model cache under `/models-cache` — our own emptyDir, should be fine;
`/plugin-site` in the watermark pod — also our own emptyDir; anything
*inside* the image itself vLLM might try to write to, e.g. under
`/vllm-workspace`) is writable by an arbitrary non-root UID with primary
GID 0, which is the convention Red Hat's own container images follow but
upstream `vllm/vllm-openai` was not confirmed to follow. If the pod
fails to start with a permission-denied error under `restricted-v2`
(rather than sitting in the expected `/health`-not-ready state), that's
the first thing to check — the fallback is the `anyuid` SCC (already
present per `oc get scc`), granted to the `watermark` namespace's default
service account with `oc adm policy add-scc-to-user anyuid -z default
-n watermark`, but that's a real privilege escalation and should not be
reached for without first confirming it's actually needed.

## Resource-sizing note

g5.xlarge allocatable (read from the live node,
`oc get node <gpu-node> -o json .status.allocatable`): `cpu: 3500m`,
`memory: 15031468Ki` (~14.65Gi), `nvidia.com/gpu: 1`. Both vLLM pods
request `cpu: "3"` and a `memory` limit of `12Gi` *plus* a `medium:
Memory` `/dev/shm` `sizeLimit` of `2Gi` (which counts toward the pod's
memory cgroup, per the task brief) — worst case `14Gi` against `~14.65Gi`
allocatable, leaving only ~0.65Gi headroom for node-local system pods
(NVIDIA device plugin, dcgm-exporter, etc., all visible in the node's
`nvidia.com/gpu.deploy.*` labels). This was sized exactly per the task
brief's numbers, not independently re-derived; if node-local system pods
get evicted/starved in practice, tighten the vLLM pod's memory limit
first before touching anything else.

## Image digest verification (evidence)

```
$ curl -s https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.18.0
-> HTTP 200; tag_status "active"; digest (manifest list):
   sha256:c32358ebfc115d56ade2acfdbcd00df5b115417dbd6006547c88f07e2b39de06
   images: [amd64 sha256:96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05,
            arm64 sha256:be723f3fa62508d6be28295d86de4aec9791e275474b82ed4697479242948e4d]

$ skopeo inspect --config docker://vllm/vllm-openai@sha256:96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05
-> Entrypoint: ["vllm", "serve"], Cmd: null, WorkingDir: /vllm-workspace,
   base: Ubuntu 22.04, User: (unset/root)
```

`v0.18.0` exists as a tag — no fallback-tag search was needed.
