# Fact register

Every load-bearing claim in this repo, with its verification status and source. **If a claim is not in this register with a tag, treat it as unverified.** When implementation work upgrades or invalidates a fact (e.g., something moves from STATIC to EXECUTED, or an EXECUTED test contradicts a STATIC claim), update this file in the same commit.

## Verification tags

| Tag | Meaning |
|---|---|
| `OJ-VERBATIM` | Quote/date checked word-for-word against the authentic Official Journal / official Commission PDF |
| `OFFICIAL-SRC` | Confirmed from an official source (vendor docs, GitHub releases/API, official release notes) fetched 2026-08-07/08 |
| `EXECUTED` | We ran code and observed the result; command + raw output preserved |
| `STATIC` | Verified by reading source code — **not** by execution |
| `CORROBORATED` | Multiple independent secondary sources agree; primary not directly fetched |
| `OPEN` | Unverified or unresolved; must not be relied on |

---

## A. Regulatory

| # | Fact | Tag | Source |
|---|---|---|---|
| A1 | Art. 50(2) requires providers of generative AI systems (text included) to mark outputs "in a machine-readable format and detectable as artificially generated", qualified by technical feasibility, content-type limits, cost, and state of the art | `OJ-VERBATIM` | [quotes → Art. 50(2)](quotes.md#art-50-2) |
| A2 | Article 50 applies since **2 August 2026** (Art. 113 general application; Chapter IV in no exception list) | `OJ-VERBATIM` | AI Act Art. 113 |
| A3 | New Art. 111(4) (added by Reg. 2026/1744) gives systems **"placed on the market before 2 August 2026"** until **2 December 2026** to comply with Art. 50(2) | `OJ-VERBATIM` | [quotes → Art. 111(4)](quotes.md#art-111-4), [OJ snapshot](../research/sources/omnibus-32026R1744.html) |
| A4 | The grace period's operative text omits "or put into service", while Art. 111(2) in the same amending point uses "placed on the market or put into service" — internal-only systems may not qualify | `OJ-VERBATIM` (text) / `OPEN` (legal conclusion — counsel) | [quotes → Art. 111(4)](quotes.md#art-111-4) |
| A5 | The grace period covers **only** the Art. 50(2) marking duty; 50(1) interaction disclosure and 50(4) deployer text disclosure have applied since 2 Aug 2026 with no grace period | `OJ-VERBATIM` + Guidelines | [quotes → Art. 50(4)](quotes.md#art-50-4) |
| A6 | Fines for Art. 50 breaches: up to €15M or 3% worldwide annual turnover (Art. 99(4)(g)); SMEs pay the lower (99(6)) | `OJ-VERBATIM` | [quotes → Art. 99(4)](quotes.md#art-99-4) |
| A7 | Code of Practice (final 10 Jun 2026; ~190 signatories; Commission + AI Board adequacy-confirmed): for **free-form text**, a single imperceptible-watermark layer is required and sufficient; mandatory above **200 tokens**; sub-200-token text excepted (threshold expected to ratchet down) | `OJ-VERBATIM` (PDF) | [quotes → CoP Measure 1.1](quotes.md#cop-measure-1-1) |
| A8 | Containerised text (documents/files) requires **two** layers: digitally signed metadata + imperceptible watermark | `OJ-VERBATIM` (PDF) | [quotes → CoP Measure 1.1](quotes.md#cop-measure-1-1) |
| A9 | Logging/fingerprinting **alone** is insufficient; metadata cannot be carried by free-form text; a detection mechanism must be **available** or marking does not comply (Guidelines para 70) | `OJ-VERBATIM` (PDF) | [quotes → Guidelines para 70](quotes.md#guidelines-para-70) |
| A10 | Code signatories must implement a detection-interoperability solution by **2 February 2027** (Measure 3.4); professional-setting deployments may restrict detector access to affected persons | `OJ-VERBATIM` (PDF) | [quotes → Guidelines para 70](quotes.md#guidelines-para-70) |
| A11 | An enterprise that develops/puts into service its own gen-AI system on open weights is "provider" and "deployer" and bears Art. 50 duties; Art. 2(12) open-source exemption excludes Art. 50; reliance on upstream marking doesn't transfer responsibility | `OJ-VERBATIM` (definitions, Art. 2(12), Guidelines quotes) / interpretation flagged | [quotes → who is bound](quotes.md#who-is-bound) |
| A12 | The Code permits both "post-hoc watermarking" and "model watermarking" (decode-time is not the only legally named family — it is the practical state of the art for text); alternative techniques allowed with proven equivalence + documented internal testing (Measure 4.2) | `OJ-VERBATIM` (PDF) | [quotes → CoP Measure 1.1](quotes.md#cop-measure-1-1) |
| A13 | No enforcement action on text marking published as of 2026-08-08 (obligations days old) | `OPEN` (absence of evidence) | searches 2026-08-07 |

## B. vLLM and watermarking technology

| # | Fact | Tag | Source |
|---|---|---|---|
| B1 | vLLM has no built-in watermarking (generation or detection); the only built-in logits processors are MinP, LogitBias, MinTokens | `OFFICIAL-SRC` | `vllm/v1/sample/logits_processor/builtin.py` (main, fetched 2026-08-07) |
| B2 | No RFC/issue/PR/discussion for content watermarking exists in vllm-project as of 2026-08-07; all "watermark" hits are a KV-cache eviction parameter; `vllm-project/rfcs` is empty | `OFFICIAL-SRC` (exhaustive `gh` search) | GitHub search 2026-08-07 |
| B3 | V1 custom LogitsProcessor API: `validate_params(cls, sampling_params)`, `__init__(vllm_config, device, is_pin_memory)`, `apply(logits) -> Tensor`, `is_argmax_invariant() -> bool`, `update_state(batch_update)` | `OFFICIAL-SRC` | https://docs.vllm.ai/en/latest/features/custom_logitsprocs/ |
| B4 | Loading: `--logits-processors` (FQCN) or setuptools entry points group `vllm.logits_processors`; per-request params via `SamplingParams.extra_args` / REST `vllm_xargs` / OpenAI SDK `extra_body`; processor set immutable after engine init | `OFFICIAL-SRC` | vLLM docs custom_logitsprocs + custom_arguments |
| B5 | Extension point landed in vLLM **0.10.1** (PR #19912, RFC #17799); V0 per-request callable `SamplingParams.logits_processors` fully **removed in v0.17.0** (PR #34400) | `OFFICIAL-SRC` | GitHub PRs/tags, source diff v0.16.0→v0.17.0 |
| B6 | Latest vLLM release 2026-08-07: **v0.26.0** (2026-07-27); pins torch==2.11.0, transformers>=5.5.3 | `OFFICIAL-SRC` | GitHub releases, PyPI JSON |
| B7 | Custom logits processors are **hard-incompatible with speculative decoding** — engine raises "Custom logits processors are not supported when speculative decoding is enabled." at config validation; opt-in support is open unmerged PR #43672 | `OFFICIAL-SRC` (source string) | `vllm/v1/sample/logits_processor/__init__.py`; PR #43672 |
| B8 | Model Runner V2 doesn't support custom logits processors yet (silent fallback to V1 runner); PR #47585 open (active 2026-08-04) | `OFFICIAL-SRC` | PR #47585 |
| B9 | Structured-output grammar bitmasks apply to logits **before** logits processors run | `STATIC` | `gpu_model_runner.py`, `sampler.py` (main) |
| B10 | Tensor-parallel / prefix-caching / chunked-prefill interaction with custom logits processors: undocumented, untested by anyone found | `OPEN` | — |
| B11 | `eth-sri/unified-watermarking`: uses the current V1 plugin API (AdapterLogitsProcessor); import paths/signatures match v0.26.0; 6 schemes + `detect()` each; **no LICENSE file**; 1★, 2 commits, last push 2026-02-11; companion to arXiv 2602.06754. Detection ran standalone on CPU (KGW/Chi2/AAR clean p<1e-10; SynthID after an int/float init bug workaround) | `STATIC` (vLLM integration) + `EXECUTED` (CPU detection) | repo inspection + smoke test 2026-08-07 |
| B12 | `dapurv5/vLLM-Watermark`: monkey-patches vLLM's private `Sampler` via internal attribute traversal; offline `LLM()` only with `VLLM_ENABLE_V1_MULTIPROCESSING=0`; **no OpenAI-server path**; no per-request control; latent signature drift vs current `Sampler.forward()`; license metadata inconsistent (pyproject MIT vs LICENSE Apache-2.0); 4★; author: "not production ready" | `STATIC` | repo inspection 2026-08-07 |
| B13 | Hugging Face `transformers` (4.57.6 tested) officially ships KGW (`WatermarkingConfig`/`WatermarkDetector`) and SynthID-Text (`SynthIDTextWatermarkingConfig` + detectors), Apache-2.0 | `EXECUTED` | [research/demo](../research/demo/) |
| B14 | Local CPU proof (gpt2, 60 tokens, bias 2.5, selfhash): watermarked z=5.37/p=5.4e-9 detected; unwatermarked z=0.89/p=0.30 and human z=−0.54/p=0.58 not flagged; SynthID generation confirmed working | `EXECUTED` | [research/demo/results.md](../research/demo/results.md) |
| B15 | SynthID-Text: open-sourced by Google (Apache-2.0), the only production-deployed text watermark (Gemini); Bayesian detector needs training (~10k matched examples); simpler weighted-mean scorer available untrained; Google's 20M-response Gemini A/B: quality impact ±0.01–0.02pp on thumbs rates | `OFFICIAL-SRC` (repo/paper) + `CORROBORATED` (A/B numbers) | google-deepmind/synthid-text; Nature 634 (2024) |
| B16 | MarkLLM (THU-BPM): 1,000+★, Apache-2.0, 19+ algorithms, research/eval toolkit with a vLLM demo script — most-starred watermarking codebase | `OFFICIAL-SRC` | github.com/THU-BPM/MarkLLM |
| B17 | Robustness limits: recursive paraphrase drops KGW from 99.8% to 9.7% TPR@1%FPR (arXiv 2303.11156); one back-translation drops SynthID F1 1.0→0.71 (arXiv 2508.20228); retrieval-based provenance survives paraphrase (arXiv 2303.13408). One 2026 small-sample preprint (arXiv 2607.16010) reports far worse pre-attack recall (~17–30%) — outlier, wide CIs, cite with caution | `CORROBORATED` (papers) | citations in [technical.md](technical.md) |
| B18 | Watermarking needs sampling entropy: degraded/absent at temperature 0 / greedy; low-entropy output (code, JSON, function calls) is an unsolved research problem; multi-turn/agentic re-ingestion effects are an open research gap | `CORROBORATED` | citations in [technical.md](technical.md) |
| B19 | No other inference stack ships text watermarking (SGLang, llama.cpp, NIM/Triton, Bedrock, Azure OpenAI — image only for the latter two); TGI contains legacy KGW code of unconfirmed operational status | `CORROBORATED` / `OPEN` (TGI status) | vendor docs, searches 2026-08-07 |
| B20 | No named EU enterprise case study of self-hosted production text watermarking found — this would be first-of-kind | `OPEN` (absence of evidence) | searches 2026-08-07 |

## C. OpenShift AI platform

| # | Fact | Tag | Source |
|---|---|---|---|
| C1 | RHOAI 3.4 (GA) ships vLLM v0.18.0 (CUDA/ROCm/Power/Z/Spyre) and v0.17.1 (Gaudi); RHOAI 3.3: v0.13.0 (CUDA/ROCm/Gaudi), v0.10.1.1.6 (Power/Z), v0.11–0.12 (Spyre) — all ≥ the 0.10.1 plugin-API floor | `OFFICIAL-SRC` | https://access.redhat.com/articles/rhoai-supported-configs-3.x |
| C2 | Per-deployment "Additional serving runtime arguments"/env vars (dashboard → `spec.predictor.model.args`) documented since RHOAI 2.16; RHOAI 3.4 adds llm-d `LLMInferenceService` k8s `args:` (legacy `VLLM_ADDITIONAL_ARGS` still works) | `OFFICIAL-SRC` | docs.redhat.com (customizing model deployments; 3.4 release notes) |
| C3 | Duplicating a ServingRuntime and pointing it at a custom image is a documented admin procedure | `OFFICIAL-SRC` | docs.redhat.com serving guides |
| C4 | Red Hat's Container Support Policy: modified product images not supported absent a product-specific carve-out; none found for RHOAI/RHAIIS | `OFFICIAL-SRC` (policy) / `OPEN` (product-specific answer — needs PM/support escalation) | https://access.redhat.com/articles/2726611 |
| C5 | TrustyAI/FMS Guardrails Orchestrator accepts any detector implementing `POST /api/v1/text/contents`; RHOAI ships regex/PII, HAP classifier (granite-guardian-hap-38m), Presidio — **no watermark detector** | `OFFICIAL-SRC` | RHOAI 3.4 guardrails docs; trustyai-explainability/guardrails-detectors |
| C6 | No Red Hat public statement/doc/blog/release note mentions watermarking or AI-Act text marking; public Jira search returned nothing but RHOAIENG isn't anonymously browsable (weak negative) | `OPEN` (roadmap unknown — internal question) | searches 2026-08-07 |
| C7 | Decode-time watermarking is model-agnostic (operates on final logits pre-sampling); Red Hat validates Llama/Granite/Mistral/Qwen/DeepSeek families on the same vLLM stack | `OFFICIAL-SRC` (mechanism + validated models) | vLLM docs; huggingface.co/RedHatAI |
| C8 | Whether the OpenAI endpoint on RHOAI passes `vllm_xargs`/`extra_body` through KServe/Istio untouched | `OPEN` — verify in Phase 4 | — |

## D. Known gaps (honest list — close these during implementation)

| # | Gap |
|---|---|
| D1 | **No one has run watermarking end-to-end through `vllm serve`** — all plugin-path claims are STATIC. Phase 1 exists to close this. |
| D2 | No GPU/vLLM performance overhead numbers exist (the CPU ~48% figure is single-run, non-vLLM, not representative). |
| D3 | Tensor-parallel behavior of custom logits processors: untested (B10). |
| D4 | Key management design (generation, storage, rotation, per-app scoping) does not exist anywhere off the shelf. |
| D5 | TrustyAI detector-contract fit for a watermark detector is architectural inference — not yet built/tested (Phase 3). |
| D6 | Supportability of a modified runtime image on RHOAI: needs internal Red Hat PM/support answer (C4). |
| D7 | Grace-period applicability to internal-only deployments: counsel question (A4). |
| D8 | SynthID Bayesian detector training data/effort for the chosen model: unquantified until Phase 2. |
