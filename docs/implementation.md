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
- Per-request control via `vllm_xargs` (e.g. `watermark: on/off`, `watermark_key_id`) with `validate_params()` rejecting malformed args.
- A detector CLI/module using the same key(s) (port of `transformers` `WatermarkDetector` logic), independent of vLLM.

Test protocol (all generated samples through the **OpenAI-compatible server**, not offline `LLM()`):
- ≥100 watermarked and ≥100 unwatermarked generations at ~256 tokens, temperature 0.7, plus a separately sourced ≥100-sample human-text corpus.
- Report z-score distributions, TPR at the z≥4 threshold, FPR on human corpus.
- Throughput/latency vs Phase 0 baseline (same prompts, same settings).
- Negative tests: temperature 0 (measure rather than assume; the recorded KGW run is an `EXECUTED` exception to the general degradation expectation in fact B18), structured-output request (expect composition per docs/technical.md §1 ordering), spec-decode flag (expect the documented startup error, verbatim).

**Accept when:** clean statistical separation demonstrated end-to-end through `vllm serve` (KGW at 256 tokens should show z well above 4 for watermarked and ~0 for controls); overhead quantified; all results in `EXPERIMENTS.md` with commands; facts B-register updated (D1 → EXECUTED).

## Phase 2 — SynthID-Text (production candidate)

- Second `LogitsProcessor` implementing SynthID-Text tournament sampling, ported from `transformers`/`google-deepmind/synthid-text` (Apache-2.0). Note: SynthID interacts with the *sampling* step differently than pure logit-bias schemes — validate the logits-processor formulation against the reference implementation's outputs on identical seeds before trusting it.
- Detection: start with the untrained weighted-mean scorer; measure. Then decide whether Bayesian-detector training (~10k matched examples — generate them with the Phase 1 harness) is warranted; if trained, version the detector artifact with the exact generation config it matches.
- Same test protocol as Phase 1; add a KGW-vs-SynthID comparison table (detectability at 200/256/512 tokens, quality spot-check, overhead).

**Accept when:** SynthID generation+detection works through `vllm serve` with quantified reliability at the Code's quoted 200-token threshold (`OJ-VERBATIM`, [quotes](quotes.md#cop-measure-1-1)); comparison table in `EXPERIMENTS.md`; D8 closed.

## Phase 3 — Detection service + current guardrails-path confirmation (addresses D5)

- The detector service exposes the historical FMS contract, `POST /api/v1/text/contents`, and a direct endpoint returning `{z_score, p_value, verdict, key_id, detector_version}` with an Ed25519-JWS-signed result. Both were exercised on OpenShift (`EXECUTED`, fact D5 and the [Phase 3 experiment](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half)). Signing is an engineering feature here, not a claimed legal requirement (`OPEN`).
- The FMS Guardrails Orchestrator routed KGW and SynthID requests to the detector, and an upstream `nemoguardrails==0.23.0` custom output-rail action called it with fail-closed handling (`EXECUTED`, fact D5 and the [NeMo hardening transcript](../EXPERIMENTS.md#2026-08-08--nemo-poc-hardening-evidence-full-transcript-fresh-pod-pass)).
- RHOAI 3.4 labels FMS Guardrails legacy and directs users to NeMo Guardrails (`OFFICIAL-SRC`, fact C11). The RHOAI-managed `NemoGuardrails` custom-resource path, shipped version, and retention behavior remain unverified (`OPEN`, facts C11/D5); carry that product integration into Phase 4 rather than treating either executed proof as a supported production architecture.
- The detector is designed to avoid storing submitted content and its application logs contain hashes plus verdict metadata; the executed evidence is scoped to the detector logs inspected in Phase 3 (`STATIC` / `EXECUTED`, fact D5 and the [Phase 3 experiment](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half)). Whether this exact zero-retention design is a Code requirement remains `OPEN`.

**Acceptance status:** the executable half is met: FMS and upstream NeMo paths returned the expected known-watermarked and clean verdicts with raw transcripts (`EXECUTED`, fact D5). The RHOAI-managed NeMo path remains `OPEN` and moves forward with Phase 4; this phase does not establish Red Hat supportability.

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
- Compliance mapping doc: map the exact quoted Code language and sources in [`docs/quotes.md`](quotes.md) to what this implementation does (`OJ-VERBATIM` for the source text; implementation status tagged separately). The `EXPERIMENTS.md` log may contribute to required documentation only if it meets the applicable requirements; keep it audit-grade.

**Accept when:** overhead and robustness tables published; key-management and compliance-mapping docs reviewed.

---

## Out of scope (engineering cannot close)

- Legal determination of grace-period applicability to internal-only systems (D7 — counsel).
- Red Hat support-policy carve-out and roadmap statements (D6/C6 — product management).
