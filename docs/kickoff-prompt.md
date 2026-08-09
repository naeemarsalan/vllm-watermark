# Implementation pickup — Phases 4 and 5

Use this handoff only after reading [`AGENTS.md`](../AGENTS.md), then
[`README.md`](../README.md), [`facts.md`](facts.md), and
[`implementation.md`](implementation.md). `AGENTS.md` is the binding source for
verification, licensing, secrets, cluster-safety, and evidence-handling rules.

## Registered state (updated after 2026-08-09 execution)

- Phases 0–3 have executed evidence within the recorded PoC scope: KGW and
  SynthID ran through `vllm serve`; the detector ran through standalone FMS; and
  a separate upstream `nemoguardrails==0.23.0` custom action returned the
  recorded decisions (`EXECUTED`; [facts D1, D5, and D8](facts.md),
  [`EXPERIMENTS.md`](../EXPERIMENTS.md)). The former standalone/Phase 3 scope is
  historical; the current managed path is recorded separately below.
- RHOAI 3.4 labels FMS Guardrails legacy and directs users to NeMo Guardrails
  (`OFFICIAL-SRC`; [fact C11](facts.md)). The standalone FMS
  proof is retained as historical execution evidence, not as the Phase 4 target.
- The current RHOAI-managed `NemoGuardrails` custom-resource path executed with
  metadata-only broker correlation, and the one-replica synchronous,
  non-streaming D10 `N=1`/`N=5` acceptance matrix passed and then reran through
  the current detector image (`EXECUTED`; [current build-5 D10
  evidence](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
  External KServe/Istio pass-through, D6 supportability, D4 key lifecycle,
  multi-replica/global sampling, streaming/asynchronous behavior, and
  platform-wide retention remain `OPEN` (same evidence; [facts C8/D5/D6/D10](facts.md)).
- D1 and D8 are closed; D2, D3, D4, D5, D6, D7, and the open D10 boundaries
  remain as separated in the [fact register](facts.md). D9's scoped detector
  startup contract is closed after the current startup-validation rebuild and
  recovery (`EXECUTED`), while request/cache limits and generation-side bound parity
  remain `OPEN` ([current detector reconciliation](../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09)).

## Delivery target

The destination is watermark-enabled vLLM running through an actual OpenShift AI
ServingRuntime/InferenceService path, with selected generated responses validated
through the TrustyAI-compatible detector and the current RHOAI-managed guardrails
path confirmed in Phase 4 (`EXECUTED`, scoped; [current build-5 D10
evidence](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
Legacy FMS and the standalone upstream NeMo 0.23.0 PoC remain useful executed
evidence (`EXECUTED` / `OFFICIAL-SRC`; facts C11/D5), but neither substitutes
for the managed RHOAI path. External KServe/Istio pass-through and production
supportability remain `OPEN` (same evidence; facts C8/D6).

The executed D10 contract uses a fixed-frequency selector over completed
responses, synchronous blocking, one replica, and non-streaming requests;
`N=1` selects every completed response and `N=5` selects ordinals 5, 10, …
(`EXECUTED`; [current build-5 D10
evidence](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
The 20-response `N=1` and 100-response `N=5` runs covered balanced KGW/SynthID
positive and clean cases, injected retries, bounded backpressure, and metrics.
A separate controlled real-detector outage failed closed (`EXECUTED`; [earlier
outage evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
Streaming/asynchronous and multi-replica/global sampling remain `OPEN` (facts D4/D10).

## Pickup order

1. Reconcile the shared worktree and read the existing evidence before making
   changes. Preserve concurrent work.
2. Review the executed Phase 4 scope in
   [`implementation.md`](implementation.md#phase-4--rhoai-deployment-pattern-scopes-c8-informs-d6):
   the internal predictor, managed `NemoGuardrails` action, metadata-only broker,
   and D10 acceptance evidence are recorded in the [current transcript](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted).
   Keep external gateway/Istio pass-through, supportability, and retention
   conclusions `OPEN`.
3. Continue the unblocked Phase 5 work in
   [`implementation.md`](implementation.md#phase-5--benchmarks-robustness-hardening),
   including realistic-load measurements, robustness work, key-management design,
   the remaining detector/generator resource hardening, and the open D10 production
   boundaries. The scoped D9 startup and D10 single-replica synchronous acceptance
   work is already recorded as executed; do not silently broaden those claims.
4. Append every executed command and raw output to `EXPERIMENTS.md`; update
   `facts.md` in the same change when evidence changes a registered status. Do not
   upgrade a claim to `EXECUTED` from source inspection or reconstructed output.

Do not treat Phase 0–3 reruns as missing acceptance work. Re-run them only when a
Phase 4/5 change requires regression evidence, and preserve the new command and raw
output under the repository's append-only evidence rules.
