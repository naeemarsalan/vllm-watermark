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

## 2026-08-08 — CORRECTION: Phase 1 ran two KGW processor instances (effective delta ~4.0)

While wiring SynthID, we found by reading v0.18.0 source — and then REPRODUCED IN THE
SERVING POD — that `_load_custom_logitsprocs()` concatenates auto-loaded entry-point
plugins with `--logits-processors` FQCN classes **with no deduplication**:

```
>>> _load_custom_logitsprocs(["vllm_watermark.kgw.processor:KGWLogitsProcessor"])
['KGWLogitsProcessor', 'SynthIDLogitsProcessor', 'KGWLogitsProcessor']   # KGW twice
>>> _load_custom_logitsprocs([])
['KGWLogitsProcessor', 'SynthIDLogitsProcessor']                          # fix
```

Because the Phase 1 pod both installed the wheel (entry point) and passed the flag,
**every Phase 1 "watermark on" measurement ran two stacked KGW instances — effective
delta ≈ 4.0, and two apply() passes of overhead.** What this invalidates and what it
does not:

- Still valid: end-to-end mechanism proof (D1), per-request control + HTTP-400
  validation, plugin-off ≈ zero overhead, spec-decode error, all FPR numbers
  (controls had no active processor), the wheel-injection deploy flow.
- Superseded: watermarked z-distributions (were δ≈4), watermark-on overhead
  (was 2 passes), and the "temp-0 not degraded" B18 note (was at δ≈4).

Fix: manifests pass NO `--logits-processors` flag (entry points only — exactly one
instance of each processor; verified in-pod, output above). Corrected single-instance
measurements below. `docs/api-notes-vllm-v0.18.0.md` §8 documents the trap.

## 2026-08-08 — Phase 1 corrected + Phase 2 SynthID through `vllm serve` (closes D8)

**Environment:** same pod design (new GPU node `ip-10-0-0-85` after the sandbox's
external scale-down — see below), image v0.18.0 by digest, wheel with both processors
(entry points kgw + synthid), key `poc-2026-08` from Secret, VLLM_WATERMARK_DEFAULT=on,
scheme via `vllm_xargs watermark_scheme` (default kgw). SynthID: depth 30, ngram_len 5,
sampling table 2^16 seed 0, layer keys derived from the secret digest with canonical
label `vllm_watermark.synthid.core.SYNTHID_KEY_LABEL`.

### SynthID device-independence + GPU hot path (EXECUTED in pod, A10G)

- `g_values` GPU vs CPU: **0 mismatching trials / 20** (full vocab 151936, depth 30) —
  integer LCG + bit-exact moved table; CPU detection of GPU-generated text is sound.
- `process_scores_row`: **2.57 ms/call on A10G vs 290.98 ms/call CPU** (measured
  locally, clean run) — the device-native path is what makes SynthID servable
  (~113×). CPU cost is architectural: depth×vocab sequential reduction passes.

### Detection statistics (EXECUTED; 512-token corpora, T=0.7, n=120 each; controls:
120 unwatermarked + 150 human)

KGW single-instance δ=2.0 (report `benchmarks/data/phase2_kgw_fixed_report.md`):

| corpus | n | mean z | min z | TPR@z≥4 | FPR@z≥4 |
|---|---|---|---|---|---|
| KGW δ2 T=0.7 512tok | 120 | 13.41 | 5.72 | **1.000** | — |
| KGW δ2 T=0.0 256tok | 40 | 9.91 | 6.47 | **1.000** | — |
| unwatermarked | 120 | −0.08 | — | — | **0.000** |
| human | 150 | 0.03 (max 3.13) | — | — | **0.000** |

Corrected temp-0 note: at matched length (z scales ~√T; 13.41 at 512 ⇒ ~9.5 expected
at 256), temp-0's mean z 9.91 shows **no additional degradation** for KGW δ2 on this
model — the B18 literature expectation still did not materialize at the honest delta.
Quality cost of greedy+bias remains unmeasured (Phase 5).

SynthID (untrained scorers; reports `benchmarks/data/phase2_synthid_report_*.md`):

| scorer | watermarked mean z (min) | TPR@z≥4 | control FPR |
|---|---|---|---|
| mean | 22.51 (12.24) | **1.000** | **0.000** (n=270) |
| weighted mean | 23.61 (14.11) | **1.000** | **0.000** (n=270) |

**D8 decision data:** the untrained weighted-mean scorer already achieves perfect
separation at 512 tokens on this model; Bayesian-detector training (~10k matched
examples, generable with our harness in <1h GPU) is NOT required for PoC-grade
reliability. (200/256-token truncation table appended below when the comparison run
completes.)

### Overhead (EXECUTED; 100 req, 256 tok, T=0.7, conc 4, in-cluster)

| configuration | agg. tok/s | p50 | vs 904.35 baseline |
|---|---|---|---|
| both processors loaded, watermark off | 913.78 | 1.115s | ~0% |
| KGW on, single instance, LRU 1024 | **643.38** | 1.576s | **1.41×** |
| SynthID on, GPU path, depth 30 | **287.82** | 3.563s | **3.17×** |
| (superseded: KGW double-instance) | 444.83 | 2.275s | 2.03× |

### Operational note

The sandbox's external automation scaled the GPU MachineSet to 0 again during this
phase (~hourly pattern holds). The replacement node received the
`node-role.kubernetes.io/gpu` label automatically — the MachineSet fix from Phase 0
verified in production. GPU scaled to 0 at phase end (all remaining work is CPU-side).

**Phase 2 acceptance: MET** (SynthID generation+detection through `vllm serve` with
quantified reliability; comparison table + 200-token row pending the local scoring
run, appended below).

## 2026-08-08 — Phase 3: detector service + FMS GuardrailsOrchestrator end-to-end (closes D5's executable half)

**Context shift recorded first:** facts C11 (added in a parallel session, OFFICIAL-SRC)
says RHOAI 3.4 labels FMS Guardrails **legacy** and points to NeMo Guardrails. We
validated the FMS/TrustyAI detector contract anyway (it is the documented, shipped,
executable detector interface today and what Phase 3 targets), and separately
assessed the NeMo fit (`docs/api-notes-nemo-guardrails.md`): NeMo has NO first-class
external-detector interface; the pattern is a custom `@action` POSTing to any URL —
our direct endpoint fits; no watermark rail exists in the NVIDIA-NeMo org; RHOAI's
`NemoGuardrails` CR mounts arbitrary ConfigMaps (custom actions permitted at CR
schema level; live-CR verification needs RHOAI — Phase 4).

**Build (EXECUTED):** detector image built on-cluster (BuildConfig docker-strategy,
binary source from a **staged context containing only detector/ + dist/** — never the
repo root, which holds gitignored live credentials; see deploy/phase3/README.md).
Pushed: `image-registry…/watermark/detector@sha256:6250a08b…`. Deployments:
`detector` (scheme env kgw) + `detector-synthid` (scheme env synthid) — same image.
Orchestrator: `quay.io/opendatahub/ta-guardrails-orchestrator:odh-3.4.2.git`
(self-reports **fms-guardrails-orchestr8 0.16.0** on :8034/health).

**Findings by execution:**
1. This orchestrator image **silently ignores** the 0.18.3+ `path_prefix` detector
   config field (rollout succeeds, routing unchanged) — server-side scheme authority
   therefore uses two Deployments with per-Deployment `WATERMARK_DETECTOR_SCHEME`.
2. First wiring attempt failed cross-scheme because both Services selected
   `app.kubernetes.io/component: detector` (label collision → round-robin across
   schemes). Fixed with distinct component labels; endpoints verified disjoint.
3. Orchestrator forwards client `detector_params` verbatim (source-predicted,
   confirmed live: passing `{"scheme": "synthid"}` reroutes scoring).
4. Health endpoints live on the dedicated :8034 port (`/health`, `/info`), not :8033.

**Acceptance test (EXECUTED, all through `POST orchestrator:8033/api/v2/text/detection/content`,
no client scheme params — server-side authority only):**

```
[PASS] kgw     -> watermark-kgw     : detected (kgw-watermark, score 1.0)
[PASS] synthid -> watermark-synthid : detected (synthid-watermark, score 1.0)
[PASS] kgw     -> watermark-synthid : not detected      [PASS] synthid -> watermark-kgw: not detected
[PASS] clean   -> both              : not detected      [PASS] human   -> both         : not detected
both-detectors-one-request: exactly the correct detector fires with detector_id attribution
```

**Signed results (EXECUTED):** Ed25519 key from Secret; `/v1/watermark/detect`
returns detached JWS (`alg=EdDSA`, `kid=poc-signing-2026-08`, `b64=false`); verified
locally against the public key; tampered payload correctly rejected. Key material in
gitignored `cluster/`, never logged.

**Zero retention (EXECUTED):** live pod logs contain request lines and hash-based
records only — no submitted content (also pinned by a caplog unit test, 30/30 service
tests pass locally).

**Phase 3 acceptance: MET** — the orchestrator routes detection to our detector and
returns correct verdicts for known-watermarked (both schemes) and known-clean text,
end-to-end on the cluster. Remaining non-engineering item: RHOAI's FMS-legacy/NeMo
transition (C11) means the *long-term* guardrails surface needs a NeMo custom-action
integration decision — recorded in facts D5, not improvised here.

## 2026-08-08 — Phase 2 completion: KGW-vs-SynthID detectability by length (EXECUTED)

Truncation-based scoring of the 512-token corpora (each scheme scored by its own
detector; controls scored per scheme in the per-scheme reports above — FPR 0.000
everywhere). Full table: `benchmarks/data/phase2_scheme_comparison.md` (data dir
gitignored; regenerate with benchmarks/compare_schemes.py).

| tokens | KGW δ2 mean z | KGW TPR | SynthID mean z | SynthID TPR | control FPR |
|---|---|---|---|---|---|
| **200** (Code threshold) | 9.08 | **0.992** (119/120) | 13.74 | **1.000** | 0.000 |
| 256 | 10.26 | 1.000 | 15.65 | 1.000 | 0.000 |
| 512 | 14.32 | 1.000 | 22.89 | 1.000 | (n/a — controls shorter) |

Reading: SynthID (depth 30) out-detects KGW (δ2) at every length on this model, with
comfortable margin at the Code-relevant 200-token threshold. KGW's single sub-threshold
sample at exactly 200 tokens (z<4, TPR 0.992) shows its δ2 margin is thinner there —
raising δ or preferring SynthID are both available levers. Combined with zero measured
FPR (270 controls) and the overhead table above, this completes the Phase 2
acceptance packet: **Phase 2 acceptance: MET.**

## Session close-out 2026-08-08

- GPU MachineSet at 0 replicas (verified; billable node released).
- Still running on (always-on) worker nodes: detector, detector-synthid,
  orchestrator, bench pods — the validated Phase 3 stack; teardown commands in
  deploy/phase3/README.md.
- Out of scope / blocked (unchanged, per AGENTS.md §6): D6 support-policy carve-out
  (product management), D7 grace-period scope (counsel), C11 NeMo live-CR validation
  (needs RHOAI install — Phase 4).
