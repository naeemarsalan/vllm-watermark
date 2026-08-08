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
(raw JSON reproduced in the raw-evidence addendum at the end of this file).

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
the end of the 2026-08-08 work. It does not add any watermarking or serving result.

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

**Context shift recorded first:** facts C11 (added to the register the same day by separate maintainer commits, OFFICIAL-SRC)
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

**Retention posture (EXECUTED, scoped precisely):** three distinct 28-char
substrings of a submitted sample occur 0 times in the last-600-line log windows
of ALL THREE services — detector, detector-synthid, AND orchestrator (counted
python-side from `oc logs deploy/<name> --tail=600` captures so the substrings
never enter shell output; an earlier single-service 24-char/400-line count also
returned 0) — evidence of no *plaintext logging in those windows*, not a proof
of absolute zero retention. The stronger claim rests on
design + code evidence: the service writes only stdout logs (no PVC, no writable
data volume besides the tokenizer cache emptyDir), the logging paths emit
sha256-prefix + verdict only (detector/app.py, STATIC), and a caplog unit test
asserts no content reaches the log records (part of the 30/30 service suite).

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

## 2026-08-08 — Adversarial-audit resolution + raw-evidence addendum

An adversarial audit (a 6-agent verification workflow plus independent review findings) challenged the completion
claims. Every finding below gets an explicit disposition; fixes were applied in this
commit and re-tested. This addendum also supplies the raw command+output evidence
whose absence the audit correctly flagged — the earlier prose entries ("captured in
session") did not meet this repo's EXECUTED bar on their own.

### Corrections to earlier entries (append-only; earlier text stands as history)

1. **SynthID overhead multiplier**: the Phase 2 table printed **3.17×**, which mixed
   baselines (913.78/287.82). Against the table's own stated 904.35 baseline it is
   **3.14×** (904.35/287.82=3.142). KGW's 1.41× was computed correctly.
2. **Orchestrator image attribution**: the Phase 3 entry named
   `quay.io/opendatahub/ta-guardrails-orchestrator:odh-3.4.2.git` — that was the
   research candidate from the contract survey, NOT what deploy/phase3/orchestrator.yaml
   pins and what actually ran. Deployed (verified live against the Deployment spec):
   `quay.io/trustyai/ta-guardrails-orchestrator@sha256:f3952b13…`, self-reporting
   `{"fms-guardrails-orchestr8":"0.16.0"}` — which also corrects the manifest header's
   own 0.17.0 release-date inference. All conclusions stand (0.16.0 likewise predates
   the 0.18.3 `path_prefix` field; its silent-ignore was verified live either way).
3. **Structured-output magnitude is δ4-era**: the guided_json test (8/8 detected,
   mean z 14.5) ran in the double-instance window. Composition (grammar + watermark
   coexist; output detected) is delta-independent and stands; the z **magnitude** at
   true δ2 is unmeasured (needs the GPU node scaled up again; recorded as pending work). facts.md B9
   updated accordingly.
4. **D8 time estimate**: "<1h A10G" was uncited; replaced in facts.md with arithmetic
   from measured throughput (10k×256 tok at 287.82 tok/s ≈ 2.5 h + controls ≈ 0.8 h).
5. **"512-token corpora (n=120)" phrasing**: the Phase 2 entry's tables label the
   corpora "512tok / n=120". Precisely: n=120 completions REQUESTED with
   max_tokens=512 and scored over full completions; only 76 (KGW) / 96 (SynthID)
   reach ≥512 tokenizer tokens (the comparison table's 512-truncation row uses
   exactly those subsets — its n columns are the honest lengths-reached counts).
6. **Phase 2/3 reproduction commands** — RECONSTRUCTED COMMAND SUMMARY, not raw
   execution transcripts: these are the commands executed during the 2026-08-08 work, re-assembled
   for readability (generation ran inside the bench pod via `oc exec … sh -c`
   batteries whose per-run stdout summaries are quoted in the Phase-2 sections
   above; analysis ran locally with `PYTHONPATH=src` and `WATERMARK_KEYS` set).
   Every command below is complete and literal — env prefixes included, no
   elided arguments:

   ```
   # corpora (inside the bench pod; scripts previously oc cp'd to /tmp)
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 gen_corpus.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 120 --max-tokens 512 --temperature 0.7 --watermark on \
     --key-id poc-2026-08 --scheme kgw --out corpus_kgw512_fixed.jsonl
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 gen_corpus.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 120 --max-tokens 512 --temperature 0.7 --watermark on \
     --key-id poc-2026-08 --scheme synthid --out corpus_synthid512.jsonl
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 gen_corpus.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 40 --max-tokens 256 --temperature 0.0 --watermark on \
     --key-id poc-2026-08 --scheme kgw --out corpus_kgw_temp0_fixed.jsonl
   # control corpus (pre-existing input to the analyses above — generated during
   # Phase 1, before the --scheme flag existed; flag-free form still valid):
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 gen_corpus.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 120 --max-tokens 256 --temperature 0.7 --watermark off \
     --out corpus_wm_off.jsonl
   # human corpora (local)
   PYTHONPATH=src /usr/bin/python3 benchmarks/fetch_human_corpus.py --chunk-tokens 512 \
     --n 150 --out benchmarks/data/human_corpus_512.jsonl
   # serving benchmarks (inside the bench pod; one command per configuration)
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 bench_serving.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
     --extra-body '{"vllm_xargs": {"watermark": "on", "watermark_key_id": "poc-2026-08", "watermark_scheme": "kgw"}}' \
     --out bench_kgw_fixed.json
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 bench_serving.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
     --extra-body '{"vllm_xargs": {"watermark": "on", "watermark_key_id": "poc-2026-08", "watermark_scheme": "synthid"}}' \
     --out bench_synthid.json
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 bench_serving.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
     --extra-body '{"vllm_xargs": {"watermark": "off"}}' --out bench_off2.json
   # analysis (local)
   PYTHONPATH=src python3 benchmarks/analyze_detection.py \
     --corpus benchmarks/data/corpus_kgw512_fixed.jsonl --corpus benchmarks/data/corpus_kgw_temp0_fixed.jsonl \
     --corpus benchmarks/data/corpus_wm_off.jsonl --corpus benchmarks/data/human_corpus.jsonl \
     --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct --key-id poc-2026-08 --out benchmarks/data/phase2_kgw_fixed_report
   PYTHONPATH=src python3 benchmarks/analyze_detection.py --scheme synthid \
     --corpus benchmarks/data/corpus_synthid512.jsonl --corpus benchmarks/data/corpus_wm_off.jsonl \
     --corpus benchmarks/data/human_corpus.jsonl --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct \
     --key-id poc-2026-08 --out benchmarks/data/phase2_synthid_report
   PYTHONPATH=src python3 benchmarks/compare_schemes.py \
     --kgw-corpus benchmarks/data/corpus_kgw512_fixed.jsonl --synthid-corpus benchmarks/data/corpus_synthid512.jsonl \
     --unwatermarked-corpus benchmarks/data/corpus_wm_off.jsonl --human-corpus benchmarks/data/human_corpus_512.jsonl \
     --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct --key-id poc-2026-08 \
     --out benchmarks/data/phase2_scheme_comparison_v2.md
   ```

### Raw evidence: vLLM plugin double-load (independent re-execution)

Fresh CPU-only pod (image v0.18.0 by digest, wheel installed, no GPU — proves the
loading semantics are GPU-independent):

```
$ oc -n watermark exec loadercheck -- sh -c 'set -e
pip install -q --no-deps --target /tmp/site /tmp/vllm_watermark-0.1.0.dev0-py3-none-any.whl
PYTHONPATH=/tmp/site python3 -c "
from vllm.v1.sample.logits_processor import _load_custom_logitsprocs
old = _load_custom_logitsprocs([\"vllm_watermark.kgw.processor:KGWLogitsProcessor\"])
new = _load_custom_logitsprocs([])
print(\"flag+entrypoints:\", [c.__name__ for c in old])
print(\"entrypoints only:\", [c.__name__ for c in new])"'
flag+entrypoints: ['KGWLogitsProcessor', 'SynthIDLogitsProcessor', 'KGWLogitsProcessor']
entrypoints only: ['KGWLogitsProcessor', 'SynthIDLogitsProcessor']
```
(pod: image `vllm/vllm-openai@sha256:c32358eb…`, CPU-only, wheel `oc cp`'d in with
its canonical filename — pip rejects renamed wheels, a failure this capture hit and
fixed on the way.)

### Raw evidence: Phase 3 verdict matrix, signing, retention, health (fresh re-run)

Produced by the COMMITTED script `benchmarks/phase3_raw_capture.py` (run per its
header: `oc cp` script + samples into the bench pod, `oc -n watermark exec bench --
python3 /tmp/raw_capture.py`). The `$ POST …` lines inside the transcript are the
script's own request labels, not shell commands — the executable command is the one
`oc exec` line above. Only echoed text content is elided, marked with its sha256
prefix:

```
### Raw orchestrator verdict matrix (server-side scheme authority; no client scheme params)

$ POST /api/v2/text/detection/content  content=<kgw sample 1> detectors=watermark-kgw
HTTP 200 -> {"detections": [{"detection": "kgw-watermark", "detection_type": "watermark", "detector_id": "watermark-kgw", "end": 1113, "metadata": {"detector_version": "vllm-watermark-detector/0.1.0.dev0", "gamma": 0.25, "key_id": "poc-2026-08", "num_green": 85, "num_tokens_scored": 188, "p_value": 7.750825489048927e-11, "scheme": "kgw", "z_score": 6.400354600105544}, "score": 0.9999999999224918, "start": 0, "text": "[TEXT ELIDED, sha256:45fa45bf2b4f324c]"}]}

$ POST /api/v2/text/detection/content  content=<kgw sample 1> detectors=watermark-synthid
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<synthid sample 1> detectors=watermark-kgw
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<synthid sample 1> detectors=watermark-synthid
HTTP 200 -> {"detections": [{"detection": "synthid-watermark", "detection_type": "watermark", "detector_id": "watermark-synthid", "end": 1527, "metadata": {"depth": 30, "detector_version": "vllm-watermark-detector/0.1.0.dev0", "key_id": "poc-2026-08", "mean_g": 0.5726467331118494, "num_tokens_scored": 301, "p_value": 8.620629879501494e-51, "scheme": "synthid", "score": 0.5875031677758222, "scorer": "weighted_mean", "z_score": 14.943229604963577}, "score": 1.0, "start": 0, "text": "[TEXT ELIDED, sha256:7fc176c179a6ff71]"}]}

$ POST /api/v2/text/detection/content  content=<clean sample 1> detectors=watermark-kgw
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<clean sample 1> detectors=watermark-synthid
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<human sample 1> detectors=watermark-kgw
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<human sample 1> detectors=watermark-synthid
HTTP 200 -> {"detections": []}

### Raw signed direct-endpoint response (kgw sample 1)
HTTP 200 -> {"detector_version": "vllm-watermark-detector/0.1.0.dev0", "key_id": "poc-2026-08", "model_tokenizer": "Qwen/Qwen2.5-0.5B-Instruct", "num_tokens_scored": 188, "p_value": 7.750825489048927e-11, "scheme": "kgw", "scheme_details": {"gamma": 0.25, "num_green": 85}, "score": 0.9999999999224918, "signature": "eyJhbGciOiJFZERTQSIsImI2NCI6ZmFsc2UsImNyaXQiOlsiYjY0Il0sImtpZCI6InBvYy1zaWduaW5nLTIwMjYtMDgiLCJ0eXAiOiJKT1NFIn0..XL4-t0GMqbBIc-czGWrBuSnh4-rBgLB2Bpo-kRTHY8gt3l-6wm3VNK9DrcWw-lFkZNM3QrxSRZsKLWVTF893Dw", "signing": "enabled", "verdict": true, "z_score": 6.400354600105544}

### Orchestrator health (dedicated port 8034)
GET :8034/health -> 200 {"fms-guardrails-orchestr8":"0.16.0"}
GET :8034/info -> 200 {"services":{"watermark-synthid":{"status":"HEALTHY"},"watermark-kgw":{"status":"HEALTHY"}}}
### Zero-retention raw check (distinctive 24-char substring of the submitted kgw sample, counted in detector logs — value itself never printed)
$ oc logs deploy/detector --tail=400 | grep -cF "<24-char sample substring>"
0

### Fresh JWS verification (local, public key from gitignored cluster/)
decode_complete OK: header = {"alg": "EdDSA", "b64": false, "crit": ["b64"], "kid": "poc-signing-2026-08", "typ": "JOSE"}
tampered payload -> InvalidSignatureError (correctly rejected)

### GPU MachineSet raw
$ oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a
NAME                          DESIRED   CURRENT   READY   AVAILABLE   AGE
ocp-ai-p9j4n-gpu-us-east-1a   0         0                             10h
```

### Phase 2 reference-validation evidence (audit: "identical seeds" criterion)

The acceptance criterion "validate the logits-processor formulation against the
reference implementation's outputs on identical seeds" is satisfied by
tests/test_synthid_equivalence.py, executed locally:

```
$ PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_synthid_equivalence.py -q --durations=10
19 passed in 662.90s (0:11:02)
```

Specifically: `test_process_scores_row_matches_reference_call[50257]` and `[1000]`
drive transformers' own stateful `SynthIDTextWatermarkLogitsProcessor.__call__` and
ours on identical configs/inputs (max abs diff 0.0 over the trial set; development
probes additionally verified `_compute_keys`-fold equivalence and characterized the
reference's first-`ngram_len`-calls zero-context quirk — see the test file's
docstrings). KGW equivalence: tests/test_kgw_equivalence.py (`6 passed in 405.03s`),
exact greenlist set equality vs transformers at vocab 50257 and 151936.

### Phase 2 quality spot-check (exploratory fluency proxy — labeled precisely)

Committed script `benchmarks/quality_spotcheck.py` (Qwen2.5-0.5B, float32 CPU;
first 256 completion tokens; two estimators — UNCONDITIONAL completion PPL and
PROMPT-CONDITIONED completion PPL with prompt labels masked to −100). Selection is
a PAIRED-PROMPT selection: the first-15 prompts of the three corpora are
byte-identical (verified by SHA-256 over the prompt lists — `545179464ce6c394`
for all three; the script self-reports this check on every run). The reported
statistics are nonetheless UNPAIRED per-corpus aggregate means — no per-prompt
differencing — so between-corpus comparisons carry more variance than a paired
estimator would. An earlier single-estimator run (unconditional-only) reported
4.97/7.69/2.92; the recorded run below supersedes it.

```
$ PYTHONPATH=src /usr/bin/python3 benchmarks/quality_spotcheck.py \
    --corpus kgw_wm_d2=benchmarks/data/corpus_kgw512_fixed.jsonl \
    --corpus synthid_wm=benchmarks/data/corpus_synthid512.jsonl \
    --corpus unwatermarked=benchmarks/data/corpus_wm_off.jsonl
kgw_wm_d2      n=15 PPL uncond mean=   4.70  prompt-conditioned mean=   3.74  distinct-1=0.389 distinct-2=0.822
synthid_wm     n=15 PPL uncond mean=   7.15  prompt-conditioned mean=   5.70  distinct-1=0.346 distinct-2=0.799
unwatermarked  n=15 PPL uncond mean=   2.89  prompt-conditioned mean=   2.32  distinct-1=0.432 distinct-2=0.864
```

Reading, precisely scoped: this is an EXPLORATORY fluency proxy under one small
model at small n — not a human quality rating and not claimed as a bound on one;
PPL under the generating model partly re-measures the watermark's own
distribution shift by construction. Observed: prompt-conditioned completion PPL
rises 2.32 → 3.74 (KGW δ2) and → 5.70 (SynthID depth-30); lexical diversity dips
modestly (distinct-2 0.864 → 0.822/0.799 — NOTE: in the recorded run distinct-n
was computed over FULL completions while PPL used the first 256 tokens; the
committed script now computes both over the identical first-256-token slice for
future runs). Because the recorded PPL output above predates the script's
pairing-check print, the pairing verification was executed SEPARATELY against
the current source:

```
$ PYTHONPATH=src /usr/bin/python3 benchmarks/quality_spotcheck.py \
    --corpus kgw_wm_d2=benchmarks/data/corpus_kgw512_fixed.jsonl \
    --corpus synthid_wm=benchmarks/data/corpus_synthid512.jsonl \
    --corpus unwatermarked=benchmarks/data/corpus_wm_off.jsonl \
    --verify-pairing-only
prompt-set sha256/16 per corpus: {'kgw_wm_d2': '545179464ce6c394', 'synthid_wm': '545179464ce6c394', 'unwatermarked': '545179464ce6c394'} -> PAIRED (byte-identical prompts)
PAIRED
``` Human-rated evaluation remains Phase 5
work; no external study is cited as covering this model/configuration.

### Phase 1 z-score distribution (audit: distributions, not just summary stats)

Corrected single-instance KGW δ2 corpus: n=120 completions generated with
`max_tokens=512` at T=0.7 and scored over each FULL completion (variable length —
only 76/120 reach ≥512 tokenizer tokens; the rest stopped earlier), 20-bin
histogram from `benchmarks/data/phase2_kgw_fixed_report.md` (regenerate with the
full analyze_detection.py invocation from the reproduction-commands list above —
the same four-corpus command that produced the report):

```
[   5.72,    6.47)  ####                                      2
[   6.47,    7.22)                                            0
[   7.22,    7.97)                                            0
[   7.97,    8.72)  ########                                  4
[   8.72,    9.47)  ######                                    3
[   9.47,   10.22)  ##########                                5
[  10.22,   10.97)  #############                             7
[  10.97,   11.72)  #############                             7
[  11.72,   12.48)  #####################                     11
[  12.48,   13.23)  #############################             15
[  13.23,   13.98)  ########################################  21
[  13.98,   14.73)  ###############                           8
[  14.73,   15.48)  ###################                       10
[  15.48,   16.23)  #######################                   12
[  16.23,   16.98)  ###########                               6
[  16.98,   17.73)  ##########                                5
[  17.73,   18.48)  ##                                        1
[  18.48,   19.24)  ####                                      2
[  19.24,   19.99)                                            0
[  19.99,   20.74)  ##                                        1
```

### Phase 1 "~256 tokens" single-instance dataset (audit item: explicit result)

The corrected primary corpus is 512-token; the protocol says ~256. Disposition: for
an autoregressive decode-time watermark, the first 256 tokens of a 512-token
generation are distributionally identical to a native 256-token generation (per-step
bias, no lookahead), so the 256-token truncation row of the corrected corpus is a
valid ~256-token single-instance result: **n=116 (≥100), TPR 1.000, mean z 10.26;
control FPR 0.000 at the same 256-token row on n=115 unwatermarked + n=149 human =
264 controls** (the 269-control figure belongs to the 200-token row: 119+150). The single-instance T=0 dataset
(n=40, REQUESTED max_tokens=256 — 38/40 reached 256 completion tokens, min 192,
mixed length/stop finish reasons; mean z 9.91, TPR 1.000) corroborates at
approximately-256 native generation length.

### Phase ordering + same-commit discipline (audit item)

- **Workflow violation, recorded:** docs/implementation.md says work phases in
  order and meet acceptance before moving on. In fact, commit 59b03ee declared
  "Phase 2 acceptance: MET" while the KGW-vs-SynthID comparison table was still
  computing (its completion landed in the NEXT commit, 0701fc9), and Phase 3 was
  executed and committed in 59b03ee itself — i.e. before Phase 2's acceptance
  evidence was complete. The independently produced Phase 2 and Phase 3 results
  remain valid (each carries its own evidence), but the phase-ordering discipline
  was breached and this line is the record of it.
- facts.md tag upgrades landed in the same commits as their EXPERIMENTS.md evidence
  with one exception the audit found: fact C9 (Phase 0 baseline) was added one
  commit after its evidence (76c196e → 81e11a7). Recorded as a process violation;
  not retroactively fixable without history rewrite (declined — see below).

### Repo-policy dispositions (maintainer-level decisions, recorded not improvised)

- **Commit-trailer attribution vs AGENTS.md §4 — exact state**: origin/main
  contains 4 pushed commits, all carrying the AI attribution trailer (ef3b0e4,
  76c196e — the Phase 0 pair; 5db603d, eb4c76a — separate same-day maintainer
  commits). Additionally origin/main..HEAD holds 4 LOCAL, UNPUSHED commits
  (81e11a7, 3ae2f35, 59b03ee, 0701fc9), also with trailers — these are NOT pushed.
  Recorded disposition: branch history is preserved as-is (no rewrite/rebase);
  new commits omit the trailer; curing the already-pushed trailers would require
  a maintainer-authorized history rewrite + force-push.
- **Sandbox base domain in pushed history**: commit 5db603d redacted it from the
  working tree, but it exists in the 76c196e blob (pushed). Current tracked files
  verified clean (`git grep sandbox…` = 0 hits). Cure requires history rewrite +
  force-push (a maintainer decision) or accepting sandbox credential rotation.
- **Stale committed docs/technical.md + docs/openshift-ai.md** (still describe D1 as
  open): corrected versions exist as pending uncommitted working-tree edits,
  which this work deliberately does not touch. Cure: committing those pending
  corrections (a maintainer action).

### Combined test-suite invocation (audit: reproducible single command)

`tests/` and `detector/tests/` initially could not be collected together (two
`conftest.py` files + `from conftest import …` module imports → collection errors,
reproduced before fixing). Fixed via `--import-mode=importlib` in pyproject
`[tool.pytest.ini_options]` (with `testpaths`) and canonical-path imports in the two
test files. Reproducible invocation and result:

```
$ /usr/bin/python3 -m pytest -q
154 passed, 192 warnings in 1746.78s (0:29:06)
```
(192 warnings are third-party FastAPI/Starlette deprecations under Python 3.14.)

### NeMo Guardrails forward-path validation (upstream library — precisely scoped)

EXECUTED on-cluster (namespace watermark, one ConfigMap + one temporary pod, GPU at
0 throughout): a custom output-rail action POSTing the bot message to the detector
service blocked a known-KGW sample and passed a human sample, consistently across
three angles — direct detector call (`verdict=true, z=12.817, p=6.57e-38` vs
`verdict=false, z=0.713`), the library Python API (`check_async` →
`BLOCKED`/`PASSED`), and `nemoguardrails server` `POST /v1/checks`
(`{"status":"blocked","rail":"watermark check"}` / `{"status":"passed","rail":null}`),
nemoguardrails==0.23.0. Config + full header documentation:
`deploy/phase3/nemo-guardrails-poc.yaml`.

**Scope limits (explicit):** this validates the UPSTREAM library path only. It does
NOT prove the RHOAI-managed `NemoGuardrails` CR path (no RHOAI/operator/CRD was
installed) and does not close that part of D5. Version skew risk recorded: RHOAI's
shipped NeMo version is unverified vs the pip-installed 0.23.0, and real behavioral
differences were found between 0.23.0 and the develop-HEAD sources the earlier
api-notes cited.

**Findings of independent value:** NeMo's own defaults conflict with a
zero-retention posture (422 validation errors echo the full request body; the
server's internal event log records full message content) — a gap to close before
adopting this path for compliance use.

**Incident record:** during initial validation, a debugging step briefly echoed
non-sensitive, repo-resident benchmark text (a Gutenberg excerpt and one generated
sample) into command output, contrary to the content-handling rule for this work;
corrected immediately, the pod was deleted, nothing persisted on-cluster. A
follow-up hardening pass (fail-closed failure policy, pinned installs, env-sourced
key id, neutral header wording) was dispatched and its results are recorded below
when complete.

### Namespace-state re-verification (fourth audit round)

An independent check initially reported namespace `watermark` empty of
workloads. Direct re-verification could not reproduce that, and a follow-up
JSON-output query on the intended kubeconfig subsequently CONFIRMED the five
pods and three Available Deployments — the earlier empty result was a
display/query anomaly, not evidence of a different cluster. The block below is
an EXECUTED SUMMARY (condensed from raw output; server URL redacted per repo
convention). Exact commands: `oc get ns watermark`; `oc -n watermark get
pods,deploy,svc,cm,secrets`; `oc -n openshift-machine-api get machineset
ocp-ai-p9j4n-gpu-us-east-1a`; the health/detection lines ran via `oc -n
watermark exec bench -- python3 -c '<urllib POST>'` with the request bodies
shown:

```
### Namespace inventory re-verification, 2026-08-08T06:19:20Z
watermark   Active   4h36m
pods: bench, detector-1-build(Completed), detector-…, detector-synthid-…, orchestrator-…  (all Running)
deploy: detector 1/1, detector-synthid 1/1, orchestrator 1/1
svc: detector, detector-synthid, orchestrator, vllm
cm: orchestrator-config, nemo-watermark-config, detector-1-* (build), …
secret: watermark-key, detector-signing-key
machineset ocp-ai-p9j4n-gpu-us-east-1a: 0 replicas
### Health + one KGW + one SynthID orchestrator request, 2026-08-08T06:19:24Z
GET :8034/health -> {"fms-guardrails-orchestr8":"0.16.0"}
GET :8034/info   -> {"services":{"watermark-synthid":{"status":"HEALTHY"},"watermark-kgw":{"status":"HEALTHY"}}}
kgw sample2 -> watermark-kgw: HTTP 200 [{"detection": "kgw-watermark", "detector_id": "watermark-kgw", "score": 1.0}]
synthid sample2 -> watermark-synthid: HTTP 200 [{"detection": "synthid-watermark", "detector_id": "watermark-synthid", "score": 1.0}]
```

The recent event log shows only this work's own deployment rollouts (the
SIGNING_KEY_ID Secret-sourcing change) — no mass deletion. The contrary observation
remains unexplained from here (possible different kubeconfig/context or transient
view); all live-state claims in this log are therefore tied to explicit timestamps,
and the stack is reproducible from deploy/phase3/ regardless.


### Correction to "Session close-out" (append-only)

The earlier close-out entry listed detector, detector-synthid, orchestrator, and
bench as "still running". That statement was true when written but is superseded
as a live claim: pods have since been REPLACED by rollouts (Secret-sourced
SIGNING_KEY_ID), transient single-purpose pods existed briefly and were deleted
(`loadercheck` — the CPU loader-repro pod above — and `nemo-test`), and the
namespace-state re-verification section above (timestamped 06:19Z) is the current
evidence of record. The GPU MachineSet has remained at 0 replicas since the
close-out. Live-state statements in this file are valid only at their recorded
timestamps; the deploy/phase3 manifests are the reproducible source of truth.

## 2026-08-08 — NeMo PoC hardening evidence (full transcript, fresh-pod pass)

The section below is the verbatim command+output transcript produced in ONE
coherent fresh-pod sequence (content redacted only via [TEXT ELIDED sha256:…]
markers), covering: constrained install + zero-diff freeze check; fail-closed
on malformed-200 responses (missing verdict, non-boolean verdict); the
adversarial log-tunneling case (detector returns verdict=true plus 5 poisoned
string fields — all marker counts 0 in 135 lines of process output; the single
emitted log line carries request-local scheme/key_id and <invalid:str> markers
only); missing-vs-empty bot_message semantics; real-detector happy-path
regression; pod cleanup; GPU MachineSet at 0. The committed actions.py copy
was extracted from the final ConfigMap and py_compile-verified.

# NeMo Guardrails PoC — full evidence transcript

Date: 2026-08-08. Repo: `/home/anaeem/vllm-watermark`. Cluster: `ocp-ai`,
`KUBECONFIG=cluster/auth/kubeconfig` (never printed). Namespace: `watermark`.
Cluster objects touched in this pass: ConfigMap `nemo-watermark-config`
(updated in place) and Pod `nemo-test` (created fresh, deleted at end).
No RHOAI, operator, or CRD installed. GPU MachineSet
`ocp-ai-p9j4n-gpu-us-east-1a` untouched throughout (confirmed 0/0 before
and after, item 9 below).

This is the raw, content-redacted evidence record backing
`deploy/phase3/nemo-guardrails-poc.yaml`'s header claims, per AGENTS.md's
verification discipline ("Works" means EXECUTED — command + raw output).
All nine items below were run in one coherent sequence, in the order
listed, against a single fresh `nemo-test` Pod (items 1–7), then cleanup
(items 8–9). Submitted/returned TEXT content is never printed anywhere in
this transcript; where a command's semantics involve real corpus text, the
text is redacted as `[TEXT ELIDED sha256:<prefix>]` and the actual script
reads the text from a file at runtime rather than inlining it on any
command line.

Local corpus samples used (read locally, `oc cp`'d into the pod as files,
byte counts confirmed to match after copy, sha256 prefixes computed
locally for redaction markers only):
- `benchmarks/data/corpus_kgw512_fixed.jsonl` record[0] `text` field →
  `/tmp/kgw_sample_0.txt`, 2350 bytes, `sha256:64b551969b24`
- `benchmarks/data/human_corpus.jsonl` record[0] `text` field →
  `/tmp/human_sample_0.txt`, 1107 bytes, `sha256:d93033d39053`

---

## (1) Fresh pod creation

Command:
```
export KUBECONFIG=/home/anaeem/vllm-watermark/cluster/auth/kubeconfig
oc apply -f pod.yaml   # pod.yaml == the Pod spec below, name nemo-test
oc -n watermark wait --for=condition=Ready pod/nemo-test --timeout=90s
```

`pod.yaml` (matches the Pod spec applied; identical shape used across all
passes in this validation):
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nemo-test
  namespace: watermark
  labels:
    app: nemo-test
spec:
  restartPolicy: Never
  enableServiceLinks: false
  containers:
    - name: nemo-test
      image: python:3.12-slim
      command: ["sleep", "infinity"]
      resources:
        requests:
          cpu: 500m
          memory: 1Gi
      env:
        - name: VLLM_WATERMARK_SCHEME
          value: "kgw"
        - name: VLLM_WATERMARK_KEY_ID
          value: "poc-2026-08"
        - name: OPENAI_API_KEY
          value: "unused-model-credential-placeholder"  # never called; value chosen to not resemble a secret
      volumeMounts:
        - name: nemo-config
          mountPath: /app/config/watermark-poc
  volumes:
    - name: nemo-config
      configMap:
        name: nemo-watermark-config
```

Raw output:
```
Warning: would violate PodSecurity "restricted:latest": allowPrivilegeEscalation != false (container "nemo-test" must set securityContext.allowPrivilegeEscalation=false), unrestricted capabilities (container "nemo-test" must set securityContext.capabilities.drop=["ALL"]), runAsNonRoot != true (pod or container "nemo-test" must set securityContext.runAsNonRoot=true), seccompProfile (pod or container "nemo-test" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost")
pod/nemo-test created
pod/nemo-test condition met
```
(The PodSecurity line is an advisory warning only — the Pod was created
and became Ready. The ConfigMap `nemo-watermark-config` mounted by this
Pod already existed in the namespace with the current `actions.py`
content, applied immediately before this step from
`deploy/phase3/nemo-guardrails-poc.yaml`.)

---

## (2) Pinned install + pip freeze diff (zero differences)

Commands:
```
oc -n watermark cp deploy/phase3/nemo-poc-constraints.txt nemo-test:/tmp/nemo-poc-constraints.txt

oc -n watermark exec nemo-test -- python3 -m pip install --quiet --no-input \
  --root-user-action=ignore -c /tmp/nemo-poc-constraints.txt \
  'nemoguardrails[server]==0.23.0' httpx

oc -n watermark exec nemo-test -- python3 -m pip freeze | sort > /tmp/pass4_freeze.txt
diff /tmp/pass4_freeze.txt deploy/phase3/nemo-poc-constraints.txt
echo "diff_exit_code=$?"
```

Raw output:
```
=== install ===

[notice] A new release of pip is available: 25.0.1 -> 26.2.1
[notice] To update, run: pip install --upgrade pip
exit_code=0
=== freeze + diff ===
diff_exit_code=0
78
```
(`diff_exit_code=0` means zero differences — all 78 pinned packages,
including every transitive dependency, matched exactly between the
constraints file and this fresh Pod's actual resolved install.)

---

## (3) Malformed-200 missing-verdict → BLOCKED

An in-pod mock HTTP server (Python `http.server`, listening only inside
`nemo-test` on `localhost:9010` — the real `detector`/`detector-synthid`
Deployments were never touched or restarted) returns HTTP 200 with a
well-formed JSON object that has no `"verdict"` key at all:
```python
# mock_missing_verdict.py, served on 0.0.0.0:9010
body = {"scheme": "kgw", "key_id": "poc-2026-08"}  # no "verdict" key
```

Command:
```
oc -n watermark exec nemo-test -- sh -c '
  python3 /tmp/mock_missing_verdict.py > /tmp/mock1.log 2>&1 &
  sleep 1
  VLLM_WATERMARK_DETECTOR_URL=http://localhost:9010/v1/watermark/detect \
    python3 /tmp/test_check_generic.py 2>&1 | grep -E "^WARNING:actions|^\{"
'
```
(`test_check_generic.py` builds an `LLMRails` from `/app/config/watermark-poc`
and calls `check_async` with a synthetic, non-corpus dummy assistant
message — content is irrelevant since the mock ignores the request body
entirely.)

Raw output:
```
WARNING:actions.py:watermark_check: FAILING CLOSED (blocking response) reason=detector response missing 'verdict' field policy=closed
{"status": "RailStatus.BLOCKED", "rail": "watermark check", "blocked": true}
```

---

## (4) Malformed-200 non-boolean verdict → BLOCKED

Same mock pattern, `localhost:9011`, returning `{"verdict": "true"}` (a
JSON string, not a boolean):
```python
body = {"scheme": "kgw", "key_id": "poc-2026-08", "verdict": "true"}
```

Command:
```
oc -n watermark exec nemo-test -- sh -c '
  python3 /tmp/mock_bad_type_verdict.py > /tmp/mock2.log 2>&1 &
  sleep 1
  VLLM_WATERMARK_DETECTOR_URL=http://localhost:9011/v1/watermark/detect \
    python3 /tmp/test_check_generic.py 2>&1 | grep -E "^WARNING:actions|^\{"
'
```

Raw output:
```
WARNING:actions.py:watermark_check: FAILING CLOSED (blocking response) reason=detector response 'verdict' field is not boolean: type=str policy=closed
{"status": "RailStatus.BLOCKED", "rail": "watermark check", "blocked": true}
```

---

## (5) Adversarial marker-fields case (log-tunneling defect fix) → all marker counts 0

Same mock pattern, `localhost:9012`, `verdict: true` (a genuine bool, so
the response otherwise validates) but every OTHER field the action reads
is a distinctive marker string:
```python
PAYLOAD = {
    "verdict": True,
    "z_score": "MARKER_A_7f3d9c2e1b8a4f6d",
    "p_value": "MARKER_D_1d3b8e4a9e2a5c7f",
    "scheme": "MARKER_B_9e2a5c7f1d3b8e4a",
    "key_id": "MARKER_E_5c7f1d3b8e4a9e2a",
    "extra": "MARKER_C_3b8e4a9e2a5c7f1d",
}
```

Commands (run the rail, capture the ENTIRE process stdout/stderr to a
file, then count marker occurrences across that whole file — not just
this action's own log line, so this also covers any NeMo-internal
echoing):
```
oc -n watermark exec nemo-test -- sh -c '
  python3 /tmp/mock_adversarial.py > /tmp/mock3.log 2>&1 &
  sleep 1
  VLLM_WATERMARK_DETECTOR_URL=http://localhost:9012/v1/watermark/detect \
    python3 /tmp/test_check_generic.py > /tmp/adversarial_test_output.log 2>&1
  echo "capture_exit_code=$?"
  wc -l /tmp/adversarial_test_output.log
'

# count_markers.py: reads /tmp/adversarial_test_output.log, counts each of
# the 5 marker literals, prints ONLY the counts (must all be 0) and any
# log line matching "watermark_check verdict=" that does NOT itself
# contain any marker literal (defensive double-check before printing).
oc -n watermark exec nemo-test -- python3 /tmp/count_markers.py

# separately, confirm the rail's own decision (safe JSON line only):
oc -n watermark exec nemo-test -- sh -c 'grep -E "^\{" /tmp/adversarial_test_output.log'
```

Raw output:
```
=== run ===
capture_exit_code=0
135 /tmp/adversarial_test_output.log
=== count markers (python-side counting; only counts + sanitized line printed) ===
{"marker_counts": {"MARKER_A_ZSCORE": 0, "MARKER_B_SCHEME": 0, "MARKER_C_EXTRA": 0, "MARKER_D_PVALUE": 0, "MARKER_E_KEYID": 0}}
{"sanitized_log_lines": ["INFO:actions.py:watermark_check verdict=True z_score=<invalid:str> p_value=<invalid:str> scheme=kgw key_id=poc-2026-08"], "unsafe_lines_withheld": 0}
=== rail decision ===
{"status": "RailStatus.BLOCKED", "rail": "watermark check", "blocked": true}
```
All 5 marker counts are 0 (across the entire 135-line captured process
output, not just the action's own log line). The single log line the
action emitted logs the REQUEST-LOCAL `scheme=kgw key_id=poc-2026-08`
(the values this Pod's env actually configured), not the mock's
`MARKER_B_.../MARKER_E_...` echoes, and `<invalid:str>` markers in place
of the mock's `MARKER_A_.../MARKER_D_...` strings for `z_score`/`p_value`
— exactly the fix under test. `unsafe_lines_withheld: 0` confirms the
defensive per-line marker check never had to withhold anything (the
aggregate count and the per-line check agree).

---

## (6) bot_message missing-key → BLOCKED; empty-string → ALLOWED

Direct unit-level calls to the `watermark_check` action (imported from the
mounted ConfigMap path), no detector HTTP call reached in either case
since both short-circuit before that point:

```python
# test_botmessage2.py
r1 = await actions.watermark_check(context={"some_other_key": "x"})   # no "bot_message" key
r2 = await actions.watermark_check(context={"bot_message": ""})        # empty string
```

Command:
```
oc -n watermark exec nemo-test -- python3 /tmp/test_botmessage2.py 2>&1 \
  | grep -E "^WARNING:actions|^DEBUG:actions|^\{"
```

Raw output:
```
WARNING:actions:watermark_check: FAILING CLOSED (blocking response) reason=bot_message missing from context (integration failure) policy=closed
{"case": "bot_message_key_missing", "blocked": true}
DEBUG:actions:watermark_check: bot_message is empty -- nothing to scan, allowing.
{"case": "bot_message_empty_string", "blocked": false}
```

---

## (7) Real-detector happy-path regression (KGW → blocked, human → passed via /v1/checks)

Commands:
```
oc -n watermark exec nemo-test -- sh -c \
  'nohup nemoguardrails server --config /app/config --port 8080 --disable-chat-ui > /tmp/server.log 2>&1 & sleep 6; echo LAUNCHED'
oc -n watermark exec nemo-test -- sh -c 'grep -E "Uvicorn running|ERROR" /tmp/server.log | tail -5'

# http_checks5.py POSTs to POST /v1/checks; body constructed as:
#   {"model": "watermark-poc",
#    "messages": [{"role": "user", "content": "Please write a short paragraph."},
#                 {"role": "assistant", "content": "[TEXT ELIDED sha256:64b551969b24]"}],  # kgw sample
#                  -- or --                       {"content": "[TEXT ELIDED sha256:d93033d39053]"}]  # human sample
#    "guardrails": {"config_id": "watermark-poc"}}
# (the script reads the actual text from /tmp/kgw_sample_0.txt / /tmp/human_sample_0.txt
# at runtime -- text is never inlined on any command line or printed; the
# script's own output extracts only http_status/status/rail fields, never
# the response "content" field, on every status-code path.)
oc -n watermark exec nemo-test -- python3 /tmp/http_checks5.py
```

Raw output:
```
LAUNCHED
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
{"label": "kgw_watermarked", "http_status": 200, "status": "blocked", "rail": "watermark check"}
{"label": "human", "http_status": 200, "status": "passed", "rail": null}
```

Direct-detector ground truth for these same two samples (recorded in an
earlier pass, unchanged — `POST /v1/watermark/detect`, scheme=kgw,
key_id=poc-2026-08):
- kgw sample (`sha256:64b551969b24`): `verdict=true, z_score=12.817, p_value=6.57e-38, num_tokens_scored=400`
- human sample (`sha256:d93033d39053`): `verdict=false, z_score=0.713, p_value=0.238, num_tokens_scored=237`

---

## (8) Pod deletion + NotFound confirmation

Commands:
```
oc -n watermark delete pod nemo-test --wait=true
oc -n watermark get pod nemo-test
```

Raw output:
```
pod "nemo-test" deleted
Error from server (NotFound): pods "nemo-test" not found
```

---

## (9) GPU MachineSet confirmed at 0

Command:
```
oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a
```

Raw output:
```
NAME                          DESIRED   CURRENT   READY   AVAILABLE   AGE
ocp-ai-p9j4n-gpu-us-east-1a   0         0                             11h
```

---

## Post-cleanup state

After item 8, the `watermark` namespace's Pod list contained none of
this validation's objects (only the pre-existing `bench`,
`detector-1-build`, `detector-*`, `detector-synthid-*`, `orchestrator-*`
workloads unrelated to this validation). The ConfigMap
`nemo-watermark-config` was left in place, re-applied from
`deploy/phase3/nemo-guardrails-poc.yaml` immediately before this pass so
that the live object and the committed file are byte-identical (diffed
and confirmed as part of this same session, outside the 9 numbered items
above since it precedes item 1).


### Documentation-boundary corrections (final pass)

1. **Correction to the committed Phase 3 acceptance paragraph** (append-only): it
   called the RHOAI-managed NeMo transition a "non-engineering item". That is
   wrong — the RHOAI-managed `NemoGuardrails` CR integration (operator mount
   behavior, shipped-version verification, retention-gap mitigation) is an
   ENGINEERING integration item deferred to Phase 4; only the C11 lifecycle/
   support-posture question itself is non-engineering. facts.md D5 carries the
   precise open list.
2. **Scope of the NeMo install-reproducibility claim**: the constraints file pins
   all 78 resolved Python packages and was verified zero-drift in a fresh pod —
   that claim covers VERSION-RESOLVED PYTHON DEPENDENCIES under the recorded
   image/platform (python:3.12-slim, CPython 3.12, linux/amd64) only. The base
   image tag is mutable and no artifact hashes are pinned, so this is NOT a
   bit-reproducible container/supply-chain claim. The temporary test pod also
   ran with PodSecurity "restricted" WARNINGS (no securityContext hardening) —
   acceptable for a deleted single-purpose test pod, but any reusable deployment
   of this rail must add a restricted-profile securityContext and pin the base
   image by digest. Recorded as a limitation; no rerun claimed or needed.
3. **Evidence vs reusable artifacts**: the `/tmp/...` helper commands in the NeMo
   transcript above are EXACT HISTORICAL EXECUTION EVIDENCE from the deleted test
   pod — they are not a committed replay harness. The reusable, committed
   artifacts are `deploy/phase3/nemo-guardrails-poc.yaml` (the ConfigMap with the
   rail config and hardened actions.py) and `deploy/phase3/nemo-poc-constraints.txt`.


## 2026-08-08 — Scheme comparison v2 (per-scheme control FPR; supersedes the v1 table's control rows)

Produced by the committed, six-row-model benchmarks/compare_schemes.py (exact
invocation in the reproduction-commands list above, with
`--human-corpus benchmarks/data/human_corpus_512.jsonl`). What v2 adds over the
v1 table quoted earlier: (a) every control corpus is scored by BOTH detectors —
SynthID's FPR is now independently supported instead of implied; (b) a 512-token
human control corpus covers the 512 row (the 256-token unwatermarked corpus
still cannot — its n=0 row is explicit); (c) the config header carries the
resolved numeric depth. TPR rows are unchanged from v1 (same corpora/detectors).

# KGW vs SynthID scheme comparison

model_tokenizer=`Qwen/Qwen2.5-0.5B-Instruct` vocab_size=`151936` key_id=`poc-2026-08` gamma=`0.25` synthid_depth=`30` synthid_ngram_len=`5` truncation_lengths=`[200, 256, 512]`

## Truncation length 200 tokens

| corpus | n | mean z | rate | metric |
|---|---|---|---|---|
| kgw | 120 | 9.084 | 0.992 | TPR |
| synthid | 120 | 13.744 | 1.000 | TPR |
| unwatermarked (kgw det) | 119 | -0.148 | 0.000 | FPR |
| unwatermarked (synthid det) | 119 | -0.088 | 0.000 | FPR |
| human (kgw det) | 150 | 0.027 | 0.000 | FPR |
| human (synthid det) | 150 | -0.087 | 0.000 | FPR |

## Truncation length 256 tokens

| corpus | n | mean z | rate | metric |
|---|---|---|---|---|
| kgw | 116 | 10.261 | 1.000 | TPR |
| synthid | 117 | 15.651 | 1.000 | TPR |
| unwatermarked (kgw det) | 115 | -0.068 | 0.000 | FPR |
| unwatermarked (synthid det) | 115 | -0.060 | 0.000 | FPR |
| human (kgw det) | 150 | 0.084 | 0.000 | FPR |
| human (synthid det) | 150 | -0.070 | 0.000 | FPR |

## Truncation length 512 tokens

| corpus | n | mean z | rate | metric |
|---|---|---|---|---|
| kgw | 76 | 14.317 | 1.000 | TPR |
| synthid | 96 | 22.886 | 1.000 | TPR |
| unwatermarked (kgw det) | 0 | n/a | n/a | FPR |
| unwatermarked (synthid det) | 0 | n/a | n/a | FPR |
| human (kgw det) | 150 | 0.099 | 0.000 | FPR |
| human (synthid det) | 150 | -0.098 | 0.000 | FPR |

## Notes

- `kgw`/`synthid` rows: TPR at that scheme's own detection threshold, scored with that scheme's own detector.
- `unwatermarked (...)` / `human (...)` rows: FPR of the named detector on that control corpus -- one row per (corpus, detector) pair, so each scheme's FPR is independently supported.
- n varies by truncation length: a row is scored at length L only if it has >= L scored tokens (shorter completions are excluded, counted in n_skipped_too_short in the JSON). Control corpora generated/chunked at ~256 tokens therefore have n=0 at the 512 row unless a 512-token control corpus is supplied.
- `not present` cells mean that --*-corpus flag was omitted, not a scoring failure.

