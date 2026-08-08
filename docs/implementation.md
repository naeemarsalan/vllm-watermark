# Implementation plan

Phased, each with acceptance criteria. Work phases **in order** — each closes a gap listed in [`facts.md`](facts.md) §D. Log every run (command, environment, raw output) in `EXPERIMENTS.md` at repo root (create on first run). Update `facts.md` verification tags in the same commit as the evidence.

**Algorithm sources (license-clean):** `transformers` KGW + SynthID-Text implementations and `google-deepmind/synthid-text` (Apache-2.0), MarkLLM for reference (Apache-2.0). **Never copy from `eth-sri/unified-watermarking` (no license).**

**Environment:** OpenShift 4.20 cluster `ocp-ai` ([cluster.md](cluster.md)). GPU node is billable — `./scripts/scale-gpu.sh 1` to start work, `./scripts/scale-gpu.sh 0` when done. The local workstation cannot run vLLM (Python 3.14, no GPU) — vLLM work happens in containers/pods on the cluster.

---

## Phase 0 — Baseline serving

Stand up plain vLLM on the cluster and capture a performance baseline.

- Install OpenShift AI (RHOAI 3.4.x) or, for the first spike, run the vLLM container directly in a pod on the GPU node. Record exact image digest and vLLM version.
- Serve a small open model (e.g. `Qwen/Qwen2.5-0.5B-Instruct` or a Llama 3.x 1B–8B variant that fits a g5.xlarge / A10G 24GB).
- Baseline: tokens/sec and p50/p95 latency for a fixed prompt set (script it; keep the script in `benchmarks/`).

**Accept when:** OpenAI-compatible endpoint answers; baseline numbers recorded in `EXPERIMENTS.md`.

## Phase 1 — KGW watermark logits processor under `vllm serve` (closes D1)

Build `src/vllm_watermark/` as a pip-installable package:

- A V1 `LogitsProcessor` subclass implementing KGW green-list biasing, ported from `transformers`' implementation (Apache-2.0). `is_argmax_invariant()` → `False`. Keyed hashing seeded from a secret **read from env/mounted Secret — never hardcoded, never logged**.
- Entry point in group `vllm.logits_processors`; also loadable via `--logits-processors`.
- Per-request control via `vllm_xargs` (e.g. `watermark: on/off`, `key_id`) with `validate_params()` rejecting malformed args.
- A detector CLI/module using the same key(s) (port of `transformers` `WatermarkDetector` logic), independent of vLLM.

Test protocol (all through the **OpenAI-compatible server**, not offline `LLM()`):
- ≥100 watermarked and ≥100 unwatermarked generations at ~256 tokens, temperature 0.7, plus a ≥100-sample human-text corpus.
- Report z-score distributions, TPR at the z≥4 threshold, FPR on human corpus.
- Throughput/latency vs Phase 0 baseline (same prompts, same settings).
- Negative tests: temperature 0 (expect degraded signal — document it), structured-output request (expect composition per docs/technical.md §1 ordering), spec-decode flag (expect the documented startup error, verbatim).

**Accept when:** clean statistical separation demonstrated end-to-end through `vllm serve` (KGW at 256 tokens should show z well above 4 for watermarked and ~0 for controls); overhead quantified; all results in `EXPERIMENTS.md` with commands; facts B-register updated (D1 → EXECUTED).

## Phase 2 — SynthID-Text (production candidate)

- Second `LogitsProcessor` implementing SynthID-Text tournament sampling, ported from `transformers`/`google-deepmind/synthid-text` (Apache-2.0). Note: SynthID interacts with the *sampling* step differently than pure logit-bias schemes — validate the logits-processor formulation against the reference implementation's outputs on identical seeds before trusting it.
- Detection: start with the untrained weighted-mean scorer; measure. Then decide whether Bayesian-detector training (~10k matched examples — generate them with the Phase 1 harness) is warranted; if trained, version the detector artifact with the exact generation config it matches.
- Same test protocol as Phase 1; add a KGW-vs-SynthID comparison table (detectability at 200/256/512 tokens, quality spot-check, overhead).

**Accept when:** SynthID generation+detection works through `vllm serve` with quantified reliability at the Code-relevant 200-token length; comparison table in `EXPERIMENTS.md`; D8 closed.

## Phase 3 — Detection service + TrustyAI confirmation (closes D5)

- Wrap the detector in a service exposing:
  1. the TrustyAI detectors contract: `POST /api/v1/text/contents` (verify the exact request/response schema against the RHOAI 3.4 guardrails docs and `trustyai-explainability/guardrails-detectors` before coding — do not assume field names);
  2. a direct endpoint returning `{z_score, p_value, verdict, key_id, detector_version}` with a signed (e.g. cosign/JWS) result payload — the Code requires downloadable, digitally signed detection results for signatories.
- Deploy the GuardrailsOrchestrator on the cluster; register the watermark detector; run detection through the orchestrator against Phase 1/2 outputs.
- Zero-retention: the service must not store submitted content (Code requirement); log only hashes + verdicts.

**Accept when:** the orchestrator routes a detection request to our detector and returns a correct verdict for known-watermarked and known-clean text — demonstrated end-to-end on the cluster, transcripts in `EXPERIMENTS.md`.

## Phase 4 — RHOAI deployment pattern (closes C8, informs D6)

- Build the custom runtime image: Red Hat vLLM base + `pip install` of our package. Record base digest and Dockerfile in `deploy/`.
- Duplicate the vLLM ServingRuntime → custom image; deploy via InferenceService with `--logits-processors` in runtime args; key mounted from a Secret.
- Verify `vllm_xargs` passes through the KServe/gateway path untouched (C8).
- Document the supportability posture honestly in `deploy/README.md`; escalate the support-policy question internally (D6) — engineering cannot close that one.

**Accept when:** watermarked generation + detection runs on RHOAI proper (not a bare pod), reproducible from `deploy/` manifests.

## Phase 5 — Benchmarks, robustness, hardening

- GPU overhead at realistic batch sizes (closes D2); tensor-parallel correctness if a multi-GPU node is available (D3 — identical key/seed across ranks).
- Robustness: paraphrase and translation attacks on our own outputs (quantify; expect the literature numbers — see technical.md §3.3).
- Key management design doc (D4): generation, storage (Vault/Secrets), rotation, per-app `key_id` scoping, compromise runbook.
- Compliance mapping doc: each Code of Practice measure → what this implementation does (Measure 1.1/1.1.2, Commitment 2 access rules, Measure 3.4 interoperability by 2027-02-02, Measure 4.2 documented internal testing — the `EXPERIMENTS.md` log *is* that documentation; keep it audit-grade).

**Accept when:** overhead and robustness tables published; key-management and compliance-mapping docs reviewed.

---

## Out of scope (engineering cannot close)

- Legal determination of grace-period applicability to internal-only systems (D7 — counsel).
- Red Hat support-policy carve-out and roadmap statements (D6/C6 — product management).
