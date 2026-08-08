# EXPERIMENTS.md — append-only run log

Every entry: date, environment, command(s), raw output (condensed where huge, never
altered), and what it proves. Verification tags per [`docs/facts.md`](docs/facts.md).
Secrets are redacted as `[REDACTED]`. Newest entries at the bottom.

---

## 2026-08-08 — Phase 0 infrastructure bring-up (ocp-ai cluster)

**Environment:** OpenShift 4.20 `ocp-ai` (api.ocp-ai.<redacted-sandbox-domain>), local
workstation `oc` with `KUBECONFIG=cluster/auth/kubeconfig`.

### GPU node scale-up

```
$ ./scripts/scale-gpu.sh 1
machineset.machine.openshift.io/ocp-ai-p9j4n-gpu-us-east-1a scaled
```

NFD + NVIDIA GPU operators were already installed (from `scripts/install-gpu-operators.sh`
during provisioning):

```
$ oc get csv -n openshift-nfd
nfd.4.20.0-202607290013   Node Feature Discovery Operator   4.20.0-202607290013   Succeeded
$ oc get csv -n nvidia-gpu-operator
gpu-operator-certified.v26.3.3   NVIDIA GPU Operator   26.3.3   Succeeded
```

### Bug found + fixed: MachineSet never labels the *node* (EXECUTED)

The g5.xlarge machine reached `Running` in ~5 min and the node `ip-10-0-5-194.ec2.internal`
joined and went `Ready`, but a 30-minute wait for a node with label
`node-role.kubernetes.io/gpu` timed out. Root cause: `scripts/create-gpu-machineset.sh`
set the label only on `.spec.template.metadata.labels` (the **Machine** object), not on
`.spec.template.spec.metadata.labels` (propagated to the **Node**). Confirmed:

```
$ oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a -o jsonpath='{.spec.template.spec.metadata}'
{}
```

Fix applied (all three, same session):

```
$ oc label node ip-10-0-5-194.ec2.internal node-role.kubernetes.io/gpu="" --overwrite
node/ip-10-0-5-194.ec2.internal labeled
$ oc -n openshift-machine-api patch machineset ocp-ai-p9j4n-gpu-us-east-1a --type merge \
    -p '{"spec":{"template":{"spec":{"metadata":{"labels":{"node-role.kubernetes.io/gpu":""}}}}}}'
machineset.machine.openshift.io/ocp-ai-p9j4n-gpu-us-east-1a patched
```

plus the same line added to `scripts/create-gpu-machineset.sh` (this commit).

### GPU node verified operational (EXECUTED)

```
$ oc get node ip-10-0-5-194.ec2.internal ... (condensed)
instance-type: g5.xlarge
allocatable nvidia.com/gpu: 1
taints: None
nvidia.com/gpu.product: NVIDIA-A10G, nvidia.com/gpu.memory: 23028 (MiB)
nvidia.com/cuda.driver-version.full: 580.126.20, cuda.runtime-version.full: 13.0
```

GPU operator stack all Running on the node; `nvidia-cuda-validator` pod **Completed**
(driver validation passed).

### Serving image pinned (OFFICIAL-SRC)

Target: vLLM v0.18.0 to match RHOAI 3.4 (facts C1). Docker Hub queried 2026-08-08:

```
$ curl -s https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.18.0
name=v0.18.0
manifest-list digest: sha256:c32358ebfc115d56ade2acfdbcd00df5b115417dbd6006547c88f07e2b39de06
amd64 image digest:   sha256:96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05
```

Namespace `watermark` created; image pre-pull pod (pinned by manifest-list digest)
scheduled on the GPU node to overlap the multi-GB pull with remaining setup.

## 2026-08-08 — Phase 0 baseline serving + benchmark (EXECUTED)

**Environment:** pod `vllm-baseline` on `ip-10-0-5-194.ec2.internal` (g5.xlarge / A10G
24GB), image `vllm/vllm-openai@sha256:c32358eb…` (tag v0.18.0), engine log confirms
`Initializing a V1 LLM engine (v0.18.0)`, model `Qwen/Qwen2.5-0.5B-Instruct`
(`--max-model-len 4096 --gpu-memory-utilization 0.90`), dtype bfloat16.
Manifests: `deploy/phase0/`.

### Friction found + fixed: `VLLM_PORT` service-link collision (EXECUTED)

First start crashed at engine-core init:

```
ValueError: VLLM_PORT 'tcp://172.30.40.120:8000' appears to be a URI. This may be
caused by a Kubernetes service discovery issue,check the warning in:
https://docs.vllm.ai/en/stable/serving/env_vars.html
```

Cause: our Service is named `vllm`, so Kubernetes legacy service links inject
`VLLM_PORT=tcp://<clusterIP>:8000` into same-namespace pods, and vLLM reads
`VLLM_PORT` as its own config. Fix: `enableServiceLinks: false` on all phase0 pods
(committed). Also pre-empted: `HOME`/`XDG_CACHE_HOME` pointed at the emptyDir because
restricted-v2 SCC runs the container as an arbitrary UID with an unwritable default
`$HOME` (vLLM writes torch/triton caches under `$HOME`).

After the fix the pod went Ready in ~75s from container start (readiness probe
`/health`); `GET /v1/models` from the in-cluster bench pod returned the served model
(raw JSON captured in session log).

### Baseline benchmark (EXECUTED)

Command (in-cluster `bench` pod, python:3.12-slim + requests):

```
OPENAI_BASE_URL=http://vllm:8000/v1 python3 bench_serving.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --prompts-file prompts.txt \
  --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
  --out baseline_results.json
```

Raw summary output:

```
=== Summary ===
requests: 100 ok / 0 failed / 100 total
wall time: 28.18s
total completion tokens: 25481
aggregate output tokens/sec: 904.35
latency (s): mean=1.124 p50=1.117 p95=1.126 p99=1.430 min=1.005 max=1.441
```

Full per-request data: `benchmarks/results/phase0_baseline_results.json`.

**Phase 0 acceptance: MET.** OpenAI-compatible endpoint answers; baseline
tokens/sec + p50/p95 recorded. Comparison figures for Phase 1 overhead:
**904.35 tok/s aggregate, p50 1.117s, p95 1.126s** (concurrency 4, 256 max tokens,
temperature 0.7, prompts `benchmarks/prompts.txt`).

## 2026-08-08 — vllm_watermark package: local test suite (EXECUTED)

**Environment:** local workstation, Python 3.14.4, torch 2.9.1+cu128 (CPU),
transformers 4.57.6.

```
$ /usr/bin/python3 -m pytest tests/ -q
34 passed in 329.54s (0:05:29)
```

Covers (see tests/test_kgw_equivalence.py, tests/test_processor_static.py):
- Greenlist equivalence vs transformers' WatermarkLogitsProcessor (exact set
  equality, 200 random prev_tokens × vocab 50257 and 151936, default hash key).
- Detector z-score equivalence vs transformers' WatermarkDetector scoring logic
  (rtol 1e-6, 50 random sequences, both ignore_repeated_ngrams modes).
- Keyed generation↔detection self-consistency with a 64-bit derived hash key:
  simulated 256-step generation z > 4; random sequences z < 1.5.
- V1 processor batch bookkeeping (add/remove/move incl. swap) against a stub of the
  fetched v0.18.0 interface; apply() row-selection math on CPU tensors.

**Upstream bug found by execution (STATIC consequence for others):** in
transformers 4.57.6, `WatermarkDetector`'s `ignore_repeated_ngrams=True` is a no-op:
`_score_ngrams_in_passage` builds `collections.Counter` over `torch.Tensor` rows,
whose hash is identity-based, so identical ngrams are never merged. Our port
implements the documented semantics (value-based dedup) — see
src/vllm_watermark/kgw/detector.py DEVIATION 1.

**Pipeline smoke (EXECUTED, local, no vLLM):** simulated KGW generation (delta-biased
sampling over random logits, Qwen2.5 tokenizer decode) through
benchmarks/analyze_detection.py: watermarked n=25 mean z=27.74 TPR=1.0;
unwatermarked n=25 mean z=0.56 FPR=0.0; human corpus n=150 mean z=0.19 FPR=0.0
(report: benchmarks/data/smoke_report.md — data files gitignored). This validates the
detector/analysis tooling only — NOT `vllm serve` (that is Phase 1's job).
