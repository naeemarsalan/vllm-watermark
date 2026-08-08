# vllm-watermark

**EU AI Act Article 50(2) text watermarking for vLLM on OpenShift AI — verified research and implementation workspace.**

Goal: a decode-time text watermarking implementation (generation **and** detection) for LLMs served by vLLM on OpenShift AI, addressing the EU AI Act's machine-readable marking obligation for AI-generated text. The research base carries mixed verification tags in [`docs/facts.md`](docs/facts.md). As of 2026-08-08, KGW and SynthID-Text watermarking are `EXECUTED` end to end through `vllm serve` on the cluster (fact D1 closed; corrected single-instance statistics after the double-load finding — see the correction entry), and the watermark detector service is validated end to end through the FMS Guardrails Orchestrator, all with evidence in [`EXPERIMENTS.md`](EXPERIMENTS.md).

> This repo contains regulatory analysis but is **not legal advice**. Compliance decisions must go through counsel.

---

## Verified highlights

Every claim below carries a verification tag defined in [`docs/facts.md`](docs/facts.md), which links each fact to exact quotes in [`docs/quotes.md`](docs/quotes.md). Nothing in this repo is asserted without a stated verification status.

### The law

- **The 2 December 2026 deadline is real EU law.** Regulation (EU) 2026/1744 ("Digital Omnibus on AI", in force 27 July 2026) added Article 111(4) to the AI Act: generative AI systems *"placed on the market before 2 August 2026"* must comply with the Article 50(2) marking obligation by **2 December 2026**. Verified verbatim against the authentic Official Journal text ([quote](docs/quotes.md#art-111-4), [OJ snapshot](research/sources/omnibus-32026R1744.html)).
- **Scope caveat:** the grace period says only *"placed on the market"* — not *"put into service"*, a distinct term the same article uses two paragraphs earlier. An internal-only on-prem system may not literally qualify. This is a counsel question; the safe engineering posture is to treat Article 50(2) as due now.
- **The rest of Article 50 has applied since 2 August 2026** with no grace period — including chatbot disclosure (50(1)) and deployer disclosure of published AI text (50(4)). Fines for Article 50 breaches: up to **€15M or 3% of worldwide annual turnover** (Art. 99(4)(g)) ([quote](docs/quotes.md#art-99-4)).
- **The voluntary Code gives signatories a text-specific watermark path.** The Code of Practice on Transparency of AI-Generated Content (final 10 June 2026, ~190 signatories, confirmed adequate by the Commission and AI Board) says *"free-form text cannot transport metadata"*. Its measures treat a single watermark layer as sufficient for that channel and apply watermarking to free-form text **longer than 200 tokens**. The 200-token threshold is Code wording, not statutory text from Article 50. The Code uses a two-layer approach for exported “containerised text” ([quotes](docs/quotes.md#cop-measure-1-1)).
- **Detection is a co-equal obligation.** Commission Guidelines on Article 50, para 70: marking without an available detection mechanism *"will not suffice"* ([quote](docs/quotes.md#guidelines-para-70)). Code signatories must also make detection mechanisms interoperable by **2 February 2027**.
- **An enterprise self-hosting an open-weights model bears the obligation itself** — it is "provider" and "deployer" of its own AI system, per the Guidelines' own in-house example. The Article 2(12) open-source exemption explicitly excludes Article 50 ([quotes](docs/quotes.md#who-is-bound)).

### The technology

- **vLLM has no watermarking support and no upstream RFC for it** (exhaustively searched 2026-08-07). The correct extension point exists: the V1 custom logits-processor plugin API (`--logits-processors`, entry points, per-request `vllm_xargs`), shipped since vLLM 0.10.1 ([details](docs/technical.md)).
- **Hard constraint:** custom logits processors are incompatible with speculative decoding — vLLM errors at engine start (fix is an open, unmerged PR).
- **Preferred building blocks** (all Apache-2.0, all official):
  - Hugging Face `transformers` built-in **KGW** watermarking (`WatermarkingConfig` / `WatermarkDetector`) — fastest PoC path, detector included, fully self-contained.
  - **Google's open-sourced SynthID-Text** (`google-deepmind/synthid-text`; integrated in `transformers` ≥ 4.46) — the only text watermark running in production anywhere (Gemini); the production candidate. Its high-accuracy Bayesian detector requires training (~10k examples); a simpler weighted-mean scorer works without training.
  - **MarkLLM** (`THU-BPM/MarkLLM`, 1,000+ stars, 19+ algorithms) — the most-starred watermarking toolkit; use as algorithm reference, not as a hardened serving component.
- **Prior-art warning:** `eth-sri/unified-watermarking` matches the current vLLM V1 plugin API (statically verified against v0.26.0) and its CPU detection works (we executed it), but it has **no license — do not copy its code**; design reference only. `dapurv5/vLLM-Watermark` monkey-patches vLLM internals, offline-only, no server path — not viable for serving ([assessments](docs/technical.md#plugin-assessments)).
- **Mechanism proven locally** (CPU, transformers stack — the same logits-processor mechanism a vLLM plugin uses): watermarked output **z = 5.37** (p = 5.4e-9, detected), unwatermarked **z = 0.89**, human text **z = −0.54** (both correctly not flagged). Scripts and raw logs: [`research/demo/`](research/demo/).

### The platform

- **OpenShift AI ships new-enough vLLM.** RHOAI 3.4 (GA): vLLM 0.17.1–0.18.0; RHOAI 3.3: 0.10.1.1.6–0.13.0 — at/above the 0.10.1 plugin-API floor. Per-deployment runtime args and custom ServingRuntime images are documented flows ([details](docs/openshift-ai.md)).
- **The detection integration target must be reassessed.** The FMS/TrustyAI Guardrails Orchestrator exposes the previously selected detector API, but RHOAI 3.4 now labels FMS Guardrails legacy and directs users to NeMo Guardrails (`OFFICIAL-SRC`; fact C11). Phase 3 executed the FMS detector contract end to end (it remains the shipped, documented detector interface) and assessed the NeMo-forward surface ([api-notes](docs/api-notes-nemo-guardrails.md)); the live RHOAI `NemoGuardrails` CR check is Phase 4 work — see fact D5 for exactly what is closed vs open.
- **Supportability open item:** Red Hat's Container Support Policy does not cover customer-modified product images, and no RHOAI-specific carve-out was found. Needs product-management confirmation before any production commitment.

---

## Repo map

| Path | What it is |
|---|---|
| [`docs/facts.md`](docs/facts.md) | Fact register — every claim, its verification status, its source |
| [`docs/quotes.md`](docs/quotes.md) | Exact verbatim quotes from the legal texts, with provenance |
| [`docs/technical.md`](docs/technical.md) | vLLM extension point, plugin assessments, watermarking science — committed copy predates the Phase 1-3 execution results; see `EXPERIMENTS.md` + `docs/facts.md` for current EXECUTED state |
| [`docs/openshift-ai.md`](docs/openshift-ai.md) | OpenShift AI / TrustyAI integration facts |
| [`docs/implementation.md`](docs/implementation.md) | Phased implementation plan with acceptance criteria |
| [`docs/blog-draft.md`](docs/blog-draft.md) | Mixed-audience, evidence-annotated publication draft |
| [`docs/cluster.md`](docs/cluster.md) | The OpenShift cluster this work deploys to (GPU scale-up/down) |
| [`AGENTS.md`](AGENTS.md) | Operating rules for agents working in this repo |
| [`research/demo/`](research/demo/) | Executed CPU proof: scripts, raw logs, results |
| [`research/sources/`](research/sources/) | Snapshot of the authentic OJ text of Regulation (EU) 2026/1744 |
| `gpu/`, `scripts/`, `install-config.template.yaml` | Cluster provisioning assets (see [`docs/cluster.md`](docs/cluster.md)) |

## Status

- [x] Regulatory requirements verified against primary sources (2026-08-07/08)
- [x] Technical landscape assessed; plugin repos inspected hands-on
- [x] Watermark generate→detect mechanism proven locally (CPU, transformers)
- [x] OpenShift 4.20 cluster provisioned (`ocp-ai`; billable GPU MachineSet lifecycle documented in [`docs/cluster.md`](docs/cluster.md))
- [x] Phase 0 — baseline vLLM v0.18.0 serving and benchmark on OpenShift (`EXECUTED`; [run record](EXPERIMENTS.md#2026-08-08--phase-0-baseline-serving--benchmark-executed))
- [x] KGW package, detector, and benchmark tooling implemented (`STATIC`); 34-test local suite executed at the recorded 2026-08-08 revision (`EXECUTED`; [run record](EXPERIMENTS.md#2026-08-08--vllm_watermark-package-local-test-suite-executed))
- [x] **Phase 1 — KGW logits processor running under `vllm serve`** (2026-08-08: TPR 1.000 / FPR 0.000 end-to-end on the cluster; overhead quantified; D1 closed — see `EXPERIMENTS.md`)
- [x] Phase 2 — SynthID-Text generation + detection (2026-08-08: untrained scorers TPR 1.000/FPR 0.000 through `vllm serve`; GPU hot path 2.57ms/tok; D8 closed)
- [x] Phase 3 — Detection service validated end-to-end through the FMS Guardrails Orchestrator (2026-08-08; correct verdicts incl. cross-scheme negatives, Ed25519-signed results; retention posture: hash-only logging verified in scoped log windows plus stateless/no-data-volume design — not an absolute zero-retention claim; the NeMo library's own 422-echo/event-log retention gap is a separate, recorded open item). Guardrails-path caveat: RHOAI 3.4 marks FMS legacy (C11); the NeMo-forward surface is assessed in docs/api-notes-nemo-guardrails.md with the live RHOAI `NemoGuardrails` CR check deferred to Phase 4 — D5 records exactly what remains open.
- [ ] Phase 4 — OpenShift AI deployment (custom runtime image + ServingRuntime)
- [ ] Phase 5 — Benchmarks, robustness tests, hardening
