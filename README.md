# vllm-watermark

**Evidence-tracked text-watermarking proof of concept for vLLM on OpenShift, with EU AI Act Article 50(2) research.**

Goal: a decode-time text watermarking implementation (generation **and** detection) for LLMs served by vLLM on OpenShift AI, alongside research on the exact Article 50 marking text quoted in this repository. The research base carries mixed verification tags in [`docs/facts.md`](docs/facts.md). As of 2026-08-09, KGW and SynthID-Text ran end to end through `vllm serve`; the detector ran through standalone FMS; and the current RHOAI ServingRuntime/InferenceService/internal predictor → validation gateway → managed NeMo → authenticated broker → detector path completed the exact one-in-`N` acceptance matrix (`EXECUTED`, scoped; facts C8/D5/D9/D10; [current evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

> This repo contains regulatory analysis but is **not legal advice**. Compliance decisions must go through counsel.

## Delivery target

The destination is watermark-enabled vLLM deployed through an actual OpenShift
AI ServingRuntime/InferenceService path, with selected generated responses
validated by the TrustyAI-compatible detector through the current RHOAI-managed
guardrails path. The internal predictor and current metadata-only managed
broker/detector path are executed, including configurable one-in-`N` selection.
External KServe gateway/Istio pass-through, supportability, production network/auth
policy, multi-replica/streaming semantics, and platform-wide retention remain open
(`EXECUTED` / `OPEN`; facts C8/D5/D6/D10; [current evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
The executed standalone FMS path is legacy, and the executed upstream NeMo
0.23.0 PoC is not evidence that the managed RHOAI path works or is supported
(`EXECUTED` / `OFFICIAL-SRC` / `OPEN` as separated in facts C11/D5/D6).

Continuous validation uses strict `validation_sample_every=N`: select every `N`th
completed vLLM response at the one persistent sampler, so `N=1` validates every
completed generated-text response. The executed synchronous, non-streaming,
single-replica matrix covered `N=1` (20/20), `N=5` (20/100), both watermark schemes
and clean controls, fail-closed outage/retries, capacity-two overflow, generation/
validation/queue-lag/gateway-response-ready latency, hash-only correlation,
bounded-label metrics, and finite secret/plaintext scans. Client-observed network
delivery latency remains unmeasured
(`EXECUTED`, scoped / `OPEN`, client-observed; D10, the [current
evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted),
and the [latency semantics correction](EXPERIMENTS.md#latency-semantics-correction-2026-08-09)).

---

## Verified highlights

The active highlights below cite verification tags and sources registered in
[`docs/facts.md`](docs/facts.md). Immutable historical material can contain
unsupported or superseded context; [fact B24](docs/facts.md) defines that
boundary rather than treating it as current evidence.

### The law

- **Article 111(4) says systems *"placed on the market before 2 August 2026"* must take the necessary Article 50(2) steps by 2 December 2026** (`OJ-VERBATIM`; [quote](docs/quotes.md#art-111-4), [OJ snapshot](research/sources/omnibus-32026R1744.html), fact A3).
- **Scope remains a counsel question.** Article 111(4) says *"placed on the market"*, while Article 111(2) in the same amendment says *"placed on the market or put into service"* (`OJ-VERBATIM`). Application to an internal-only deployment is `OPEN`; no engineering conclusion is substituted for counsel (facts A4/D7).
- **Dates and penalty language are registered from quoted text.** Article 113's date, Article 111(4)'s express reference to Article 50(2), and Article 99(4)(g)'s penalty language are `OJ-VERBATIM`; application beyond that express text remains `OPEN` for counsel (facts A2/A5/A6; [Article 50(4) quote](docs/quotes.md#art-50-4), [Article 99(4) quote](docs/quotes.md#art-99-4)).
- **The voluntary Code's text-specific language is quoted, not treated as statutory wording.** It says *"free-form text cannot transport metadata"* and *"For free-form text longer than 200 tokens, watermarking still needs to be applied"* (`OJ-VERBATIM`; [Code extracts](docs/quotes.md#cop-measure-1-1), facts A7/A8).
- **Detection is part of the quoted requirement.** The Commission Guidelines say that marking *"without the means for their detection being available"* *"will not suffice"*; the Code gives the 2 February 2027 interoperability date (`OJ-VERBATIM`; [extracts](docs/quotes.md#guidelines-para-70), facts A9/A10).
- **Self-hosted open-weights application is an interpretation, not a quoted conclusion.** The definitions, in-house example, and Article 2(12) text are quoted in [who is bound](docs/quotes.md#who-is-bound); applying them to a specific enterprise deployment remains `OPEN` for counsel (fact A11).

### The technology

- **vLLM exposes the required V1 custom logits-processor API** (`OFFICIAL-SRC`, facts B3–B5). The dated upstream searches found no built-in content watermark or upstream RFC, but that absence is search-bounded and remains `OPEN` (facts B1–B2; [details](docs/technical.md)).
- **Speculative decoding is a hard constraint in the recorded stack:** source validation rejects custom logits processors with speculative decoding (`OFFICIAL-SRC`), and the vLLM 0.18.0 startup rejection was observed (`EXECUTED`, fact B7).
- **Implementation sources:**
  - Hugging Face `transformers` ships KGW generation and detection under Apache-2.0 (`OFFICIAL-SRC`, fact B13). The repository's current execution evidence is the Phase 1/2 `vllm serve` record (`EXECUTED`, facts D1/D8).
  - The Apache-2.0 SynthID-Text sources document Gemini deployment, trained Bayesian detection, and an untrained weighted-mean scorer (`OFFICIAL-SRC` / `CORROBORATED`, fact B15). The repository executed the untrained scorer through `vllm serve` (`EXECUTED`, fact D8); no claim of unique deployment or production suitability is made.
  - MarkLLM was the most-starred toolkit among repositories in the dated review and provides many algorithm references (`OFFICIAL-SRC`, fact B16); it is not used as a hardened serving component (`STATIC`, current implementation).
- **Prior-art warning:** `eth-sri/unified-watermarking` matches the current vLLM V1 plugin API (`STATIC`), but it has **no license — do not copy its code**. Its reported standalone CPU run is `OPEN`: no command or raw output was preserved in this repository, so the earlier execution claim and numbers are not evidence. `dapurv5/vLLM-Watermark` is an offline, private-internals approach (`STATIC`), not a serving implementation ([fact B11](docs/facts.md), [assessments](docs/technical.md#plugin-assessments)).
- **The historical CPU report is not registered execution evidence.** It preserves source and console text under [`research/demo/`](research/demo/), but no exact invocation/raw record in `EXPERIMENTS.md`; its numerical claims remain `OPEN` under facts B14/B24. KGW and SynthID generation/detection are instead established by the recorded Phase 1/2 serving runs (`EXECUTED`, D1/D8).

### The platform

- **OpenShift AI's documented versions meet the plugin API's version floor.** RHOAI 3.4 lists vLLM 0.17.1–0.18.0 and RHOAI 3.3 lists 0.10.1.1.6–0.13.0 (`OFFICIAL-SRC`, facts C1–C3). The custom runtime image and internal ServingRuntime/InferenceService path executed in the recorded RHOAI 3.4.2 run; product supportability remains `OPEN` (`EXECUTED` / `OPEN`, facts C4/C8/D6; [details](docs/openshift-ai.md); [recovered evidence](EXPERIMENTS.md#2026-08-08--phase-4-rhoai-exact-transcript-recovered-executed-redacted)).
- **The current detection integration is executed in scoped form.** RHOAI 3.4 labels FMS Guardrails legacy and directs users to NeMo Guardrails (`OFFICIAL-SRC`; fact C11). Phase 3 executed the standalone FMS detector path and a separate upstream NeMo 0.23.0 custom action (`EXECUTED`, fact D5; [api notes](docs/api-notes-nemo-guardrails.md)); the current RHOAI-managed `NemoGuardrails` correlation/broker action also executed through the internal gateway (`EXECUTED`, facts D5/D10; [current evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
- **Supportability remains open:** the general container policy and the dated search for a product-specific carve-out are recorded, but the RHOAI answer requires product-management/support confirmation (`OFFICIAL-SRC` / `OPEN`, facts C4/D6).

---

## Repo map

| Path | What it is |
|---|---|
| [`docs/facts.md`](docs/facts.md) | Fact register — every claim, its verification status, its source |
| [`docs/quotes.md`](docs/quotes.md) | Exact verbatim quotes from the legal texts, with provenance |
| [`docs/technical.md`](docs/technical.md) | vLLM extension point, plugin assessments, watermarking science, and links to current execution evidence |
| [`docs/openshift-ai.md`](docs/openshift-ai.md) | OpenShift AI / TrustyAI integration facts |
| [`docs/implementation.md`](docs/implementation.md) | Phased implementation plan with acceptance criteria |
| [`docs/blog-draft.md`](docs/blog-draft.md) | Mixed-audience, evidence-annotated publication draft |
| [`docs/cluster.md`](docs/cluster.md) | The OpenShift cluster this work deploys to (GPU scale-up/down) |
| [`AGENTS.md`](AGENTS.md) | Operating rules for agents working in this repo |
| [`research/demo/`](research/demo/) | Read-only historical CPU report: scripts, console text, and unregistered results |
| [`research/sources/`](research/sources/) | Snapshot of the authentic OJ text of Regulation (EU) 2026/1744 |
| `gpu/`, `scripts/`, `install-config.template.yaml` | Cluster provisioning assets (see [`docs/cluster.md`](docs/cluster.md)) |

## Status

- [x] Regulatory quotations and dates registered against cited sources (2026-08-07/08); deployment-specific legal application remains `OPEN` for counsel ([fact register](docs/facts.md), [quoted sources](docs/quotes.md))
- [x] Technical landscape assessed from the registered source review (`STATIC` / `OFFICIAL-SRC` / `OPEN` as separated in [facts B1–B20](docs/facts.md))
- [ ] Register the historical local CPU invocation and raw output before treating B14 as execution evidence (`OPEN`; [fact B14](docs/facts.md))
- [x] OpenShift 4.20 cluster provisioned (`EXECUTED`; `ocp-ai` and the recorded GPU lifecycle are documented in [`docs/cluster.md`](docs/cluster.md))
- [x] Phase 0 — baseline vLLM v0.18.0 serving and benchmark on OpenShift (`EXECUTED`; [run record](EXPERIMENTS.md#2026-08-08--phase-0-baseline-serving--benchmark-executed))
- [x] KGW package, detector, and benchmark tooling implemented (`STATIC`); 34-test local suite executed at the recorded 2026-08-08 revision (`EXECUTED`; [run record](EXPERIMENTS.md#2026-08-08--vllm_watermark-package-local-test-suite-executed))
- [x] **Phase 1 — KGW logits processor running under `vllm serve`** (`EXECUTED`, 2026-08-08: TPR 1.000 / FPR 0.000 in the registered matrix; corrected overhead quantified; [fact D1 and evidence](EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8))
- [x] Phase 2 — SynthID-Text generation + detection (`EXECUTED`, 2026-08-08: untrained scorers TPR 1.000/FPR 0.000 through `vllm serve`; GPU hot path 2.57 ms/token; [fact D8 and evidence](EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8))
- [x] Phase 3 — The detector ran end to end through standalone FMS, and an upstream NeMo 0.23.0 action separately returned the recorded block/pass decisions (`EXECUTED`, [fact D5 and evidence](EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half)). The detector's application log path is hash-based (`STATIC`), and finite checks found none of the tested distinctive sample substrings (`EXECUTED`, scoped); this is not an absolute retention claim. FMS is legacy, and NeMo 422/event-log handling remains unresolved (`OPEN`; [facts C11/D5](docs/facts.md)).
- [x] Phase 4 — RHOAI 3.4.2 operator/DSC, custom runtime image, ServingRuntime/InferenceService/internal predictor, and current managed `NemoGuardrails` metadata/broker block/pass/fail-closed flow executed (`EXECUTED`, scoped; [facts C8/D5/D10 and evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)); external gateway/Istio pass-through and D6 supportability remain `OPEN`
- [x] Phase 5 D9/D10 scope — detector startup validation was independently reviewed, corrected, rebuilt, source-matched, failed a controlled blank-value rollout, and recovered; the complete synchronous one-in-`N` matrix then reran through that current image with all 40 mode-bearing, hash-only selected-response records retained (`EXECUTED`; facts D9/D10; [current detector reconciliation](EXPERIMENTS.md#current-detector-reconciliation-2026-08-09); [current build-5 D10 rerun](EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09))
- [ ] Phase 5 remaining hardening — realistic-batch overhead, tensor parallelism, robustness, key management, compliance mapping, attempt-counter semantics, bounded broker reads, transport/schema/mismatch/cancellation coverage, detector request/cache limits, expensive maximum-config review, and generation-side bound parity remain `OPEN` ([facts D2/D3/D4/D9/D10](docs/facts.md))
