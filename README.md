# vllm-watermark

**EU AI Act Article 50(2) text watermarking for vLLM on OpenShift AI — verified research and implementation workspace.**

Goal: a decode-time text watermarking implementation (generation **and** detection) for LLMs served by vLLM on OpenShift AI, meeting the EU AI Act's machine-readable marking obligation for AI-generated text. Research is complete and verified; implementation has **not** started. Nothing here has yet been proven end-to-end on `vllm serve` — that is the first implementation milestone.

> This repo contains regulatory analysis but is **not legal advice**. Compliance decisions must go through counsel.

---

## Verified highlights

Every claim below carries a verification tag defined in [`docs/facts.md`](docs/facts.md), which links each fact to exact quotes in [`docs/quotes.md`](docs/quotes.md). Nothing in this repo is asserted without a stated verification status.

### The law

- **The 2 December 2026 deadline is real EU law.** Regulation (EU) 2026/1744 ("Digital Omnibus on AI", in force 27 July 2026) added Article 111(4) to the AI Act: generative AI systems *"placed on the market before 2 August 2026"* must comply with the Article 50(2) marking obligation by **2 December 2026**. Verified verbatim against the authentic Official Journal text ([quote](docs/quotes.md#art-111-4), [OJ snapshot](research/sources/omnibus-32026R1744.html)).
- **Scope caveat:** the grace period says only *"placed on the market"* — not *"put into service"*, a distinct term the same article uses two paragraphs earlier. An internal-only on-prem system may not literally qualify. This is a counsel question; the safe engineering posture is to treat Article 50(2) as due now.
- **The rest of Article 50 has applied since 2 August 2026** with no grace period — including chatbot disclosure (50(1)) and deployer disclosure of published AI text (50(4)). Fines for Article 50 breaches: up to **€15M or 3% of worldwide annual turnover** (Art. 99(4)(g)) ([quote](docs/quotes.md#art-99-4)).
- **For free-form text, an imperceptible watermark is required — and sufficient.** The Code of Practice on Transparency of AI-Generated Content (final 10 June 2026, ~190 signatories, confirmed adequate by the Commission and AI Board): *"free-form text cannot transport metadata"*, so a single watermark layer satisfies Article 50(2) for that channel, mandatory for free-form text **longer than 200 tokens**. Metadata-only stamping does not comply for chat/API output. Exported documents ("containerised text") need **two** layers: digitally signed metadata **and** a watermark ([quotes](docs/quotes.md#cop-measure-1-1)).
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
- **Detection belongs behind the TrustyAI Guardrails detector API** (`POST /api/v1/text/contents`) — the Guardrails Orchestrator accepts any conforming detector; no watermark detector ships today. Validating that integration is a first-class milestone ([plan](docs/implementation.md)).
- **Supportability open item:** Red Hat's Container Support Policy does not cover customer-modified product images, and no RHOAI-specific carve-out was found. Needs product-management confirmation before any production commitment.

---

## Repo map

| Path | What it is |
|---|---|
| [`docs/facts.md`](docs/facts.md) | Fact register — every claim, its verification status, its source |
| [`docs/quotes.md`](docs/quotes.md) | Exact verbatim quotes from the legal texts, with provenance |
| [`docs/technical.md`](docs/technical.md) | vLLM extension point, plugin assessments, watermarking science |
| [`docs/openshift-ai.md`](docs/openshift-ai.md) | OpenShift AI / TrustyAI integration facts |
| [`docs/implementation.md`](docs/implementation.md) | Phased implementation plan with acceptance criteria |
| [`docs/cluster.md`](docs/cluster.md) | The OpenShift cluster this work deploys to (GPU scale-up/down) |
| [`AGENTS.md`](AGENTS.md) | Operating rules for agents working in this repo |
| [`research/demo/`](research/demo/) | Executed CPU proof: scripts, raw logs, results |
| [`research/sources/`](research/sources/) | Snapshot of the authentic OJ text of Regulation (EU) 2026/1744 |
| `gpu/`, `scripts/`, `install-config.template.yaml` | Cluster provisioning assets (see [`docs/cluster.md`](docs/cluster.md)) |

## Status

- [x] Regulatory requirements verified against primary sources (2026-08-07/08)
- [x] Technical landscape assessed; plugin repos inspected hands-on
- [x] Watermark generate→detect mechanism proven locally (CPU, transformers)
- [x] OpenShift 4.20 cluster provisioned (`ocp-ai`, GPU MachineSet at 0 replicas)
- [ ] **Phase 1 — KGW logits processor running under `vllm serve`** ← next
- [ ] Phase 2 — SynthID-Text generation + detection
- [ ] Phase 3 — Detection service behind TrustyAI Guardrails API, validated end-to-end
- [ ] Phase 4 — OpenShift AI deployment (custom runtime image + ServingRuntime)
- [ ] Phase 5 — Benchmarks, robustness tests, hardening
