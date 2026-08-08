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

## 2026-08-08 — End-of-session GPU scale-down (EXECUTED)

**Environment:** OpenShift 4.20 `ocp-ai`, local `oc` with
`KUBECONFIG=cluster/auth/kubeconfig`.

The live MachineSet check showed one desired/current/ready GPU node. Per the repository's
billable-resource rule, it was scaled down before the session ended:

```
$ ./scripts/scale-gpu.sh 0
machineset.machine.openshift.io/ocp-ai-p9j4n-gpu-us-east-1a scaled

$ oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a \
    -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,CURRENT:.status.replicas,READY:.status.readyReplicas \
    --no-headers
ocp-ai-p9j4n-gpu-us-east-1a   0     0     <none>
```

This proves the billable GPU MachineSet had reached desired/current replica count zero at
the end of this session. It does not add any watermarking or serving result.

## 2026-08-08 — Evidence qualification for the local pipeline smoke

The earlier “Pipeline smoke (EXECUTED, local, no vLLM)” paragraph records only a prose
summary. Its command, raw output, and referenced gitignored report were not preserved in
this repository. Under the repository's verification rules, those numerical rates do
**not** qualify as `EXECUTED` publication evidence. Fact B22 is therefore `OPEN` until a
fresh run records its environment, command, and raw output in this append-only log. This
qualification does not alter the separately preserved 34-test suite result.

## 2026-08-08 — Phase 1: KGW watermark end-to-end through `vllm serve` (EXECUTED — closes D1)

**Environment:** pod `vllm-watermark` (manifest `deploy/phase0/vllm-watermark-pod.yaml`),
image `vllm/vllm-openai@sha256:c32358eb…` (v0.18.0), engine `v0.18.0`, model
`Qwen/Qwen2.5-0.5B-Instruct` (vocab_size 151936), A10G. Plugin injected as a wheel
(pip install --no-deps --target; PYTHONPATH), loaded via
`--logits-processors vllm_watermark.kgw.processor:KGWLogitsProcessor` (log-verified in
`non-default args`). Key: k8s Secret → `WATERMARK_KEY` env, key_id `poc-2026-08`
(64-bit hash key derived via sha256; secret material in gitignored `cluster/`, never
committed or logged). KGW params: gamma 0.25, delta 2.0, lefthash, context_width 1.
`VLLM_WATERMARK_DEFAULT=on`.

### Per-request control + validation (EXECUTED)

Through `/v1/completions` from the in-cluster bench pod:
- default (no vllm_xargs) → watermarked; `{"watermark":"off"}` → not; `{"watermark":"on","watermark_key_id":"poc-2026-08"}` → watermarked.
- `{"watermark":"banana"}` → HTTP 400 `watermark must be 'on'/'off' (or a boolean), got 'banana'`
- `{"watermark_key_idz":"x"}` → HTTP 400 `Unknown watermark_* extra_args key(s) ['watermark_key_idz']…`
- `{"watermark_key_id":"no-such-key"}` → HTTP 400 `…not found among configured keys: ['poc-2026-08']`
(validate_params → ValueError → BadRequestError chain works exactly as the STATIC
api-notes predicted.)

### Detection statistics (EXECUTED)

Corpora generated through the server (120 wm-on / 120 wm-off at temp 0.7, 40 wm-on at
temp 0.0, 256 max tokens, prompts `benchmarks/prompts.txt`), plus 150 human Gutenberg
chunks. Scored locally (CPU detector, `ignore_repeated_ngrams=True`, z≥4.0):

| corpus | n | mean z | median | min | max | TPR@z≥4 | FPR@z≥4 |
|---|---|---|---|---|---|---|---|
| watermarked (T=0.7) | 120 | 21.35 | 21.67 | 15.01 | 25.17 | **1.000** | — |
| unwatermarked (T=0.7) | 120 | −0.08 | −0.19 | −1.68 | 2.58 | — | **0.000** |
| watermarked (T=0.0) | 40 | 20.60 | 21.05 | 13.09 | 23.34 | **1.000** | — |
| human (Gutenberg) | 150 | 0.03 | −0.04 | −2.69 | 3.13 | — | **0.000** |

Full report: `benchmarks/data/phase1_report.md` (data files gitignored; regenerate via
benchmarks/gen_corpus.py + analyze_detection.py). Detector CLI spot-checks: watermarked
sample z=25.40 (p=1.2e-142), human sample z=−1.30 — JSON outputs captured above in
session; commands: `python -m vllm_watermark.cli detect --model-tokenizer
Qwen/Qwen2.5-0.5B-Instruct --key-id poc-2026-08 --file <sample> --json`.

**Negative-test surprises, recorded honestly:**
- **Temperature 0 did NOT degrade KGW at delta 2.0 on this model** (mean z 20.6 vs
  21.3 at T=0.7). Greedy argmax still flips to green wherever the model's logit margin
  < delta. The literature-derived expectation (facts B18) holds for entropy-carried
  schemes/low delta; for KGW additive bias on this small model it does not. Quality
  impact of delta under greedy decoding remains unmeasured (Phase 5).
- **Structured output composes and retains signal:** 8 guided_json completions
  (temp 0.7) all detected — mean z 14.5, min 11.2 (vs ~21 free-form). These outputs
  contained prose-heavy JSON fields; stricter low-entropy schemas will do worse (B9
  ordering unchanged: grammar mask applies before our processor).

### Spec-decode incompatibility (EXECUTED — B7 upgraded)

Pod with `--speculative-config '{"method":"ngram","num_speculative_tokens":3,
"prompt_lookup_max":4}'` + `--logits-processors …` fails at engine start:

```
(EngineCore pid=70) ERROR …     raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
(EngineCore pid=70) ValueError: Custom logits processors are not supported when speculative decoding is enabled.
```

Note: raised before the plugin module is even imported (the plugin wheel was not
installed in that pod) — the rejection precedes processor loading.

### Overhead (EXECUTED — D2 partially closed: one config measured)

Same protocol as Phase 0 (100 req, 256 tok, T=0.7, concurrency 4, in-cluster):

| configuration | agg. output tok/s | p50 | p95 | vs baseline |
|---|---|---|---|---|
| Phase 0 baseline (no plugin) | 904.35 | 1.117s | 1.126s | — |
| plugin loaded, watermark off | 914.62 | 1.114s | 1.119s | **~0% (noise)** |
| watermark on, no cache | 280.57 | 3.640s | 3.694s | **3.22× slower** |
| watermark on, LRU cache 1024 | 444.83 | 2.275s | 2.626s | **2.03× slower** |

Active-path cost is CPU `torch.randperm(151936)` (~7 ms/call measured locally,
`benchmarks/bench_greenlist.py`) once per watermarked row per decode step. The LRU
memo (pure memoization keyed `(hash_key, prev_token)` — bit-identical outputs, unit
test `test_greenlist_cache_identical_and_lru`) recovers ~59%. An 8192-entry cache
measured within noise of 1024 on partial data while the pod died mid-run (below) —
default stays 1024. Remaining overhead is a Phase 5 optimization target (candidates:
thread-pool across rows, int32 storage, pinned-memory async copies).

### Operational friction found by execution (all fixed in manifests)

1. **Liveness-probe kill under load:** with the 1s default probe timeout, sustained
   watermark-on load starved `/health` (CPU-bound greenlist work) → kubelet killed the
   pod mid-benchmark (exit 137, explicit `Killing` event). Fix: probe
   `timeoutSeconds` 5 (readiness) / 10 (liveness), liveness `initialDelaySeconds` 300.
2. **External GPU scale-down automation:** at ~03:00Z the sandbox scaled the GPU
   MachineSet to 0 (no ClusterAutoscaler/MachineAutoscaler exists; no in-cluster
   cronjob; machine-api events show drain+delete). The node — and the pod on it —
   vanished mid-session. Assume the g5 node can disappear at any hour boundary;
   re-scale with `./scripts/scale-gpu.sh 1` and re-inject. (Suspected trigger: sandbox
   cost automation, possibly keyed on the `openshift-ai-node=gpu` AWS tag.)

**Phase 1 acceptance: MET.** Clean statistical separation end-to-end through
`vllm serve` (TPR 1.000, FPR 0.000, human max z 3.13 < 4); overhead quantified
(table above); per-request control validated; negative tests executed verbatim.
