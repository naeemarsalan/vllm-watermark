# Technical landscape

Verification tags per [`facts.md`](facts.md). Fetch dates: 2026-08-07/08.

## 1. vLLM's extension point (the hook we build on)

vLLM's V1 engine exposes a documented plugin API for custom logits processors — batch-level processors that adjust the logits tensor after the model forward pass and before sampling. This is exactly where decode-time watermark bias injection belongs, and it requires no vLLM source modification. `OFFICIAL-SRC`

**Interface** (`vllm.v1.sample.logits_processor.LogitsProcessor`):

| Method | Role |
|---|---|
| `validate_params(cls, sampling_params)` | Reject bad per-request args at request time |
| `__init__(self, vllm_config, device, is_pin_memory)` | Engine-init construction |
| `apply(self, logits) -> torch.Tensor` | The hot path — adjust batch logits |
| `is_argmax_invariant(self) -> bool` | Watermarking changes argmax → must return `False` |
| `update_state(self, batch_update)` | Track per-request state across batch Add/Remove/Move |

**Loading:** `--logits-processors <FQCN>` on `vllm serve` / `LLM()`, or a setuptools entry point in group `vllm.logits_processors` (auto-discovered — a pip-installed package is enough). The processor set is immutable after engine init.

**Per-request control:** `SamplingParams.extra_args` offline; `vllm_xargs` in the REST body (OpenAI SDK: `extra_body={"vllm_xargs": {...}}`) online. This is how per-request keys / enable-disable reach the processor.

**Version facts** `OFFICIAL-SRC`:

| Event | Version | Ref |
|---|---|---|
| Extension point landed | v0.10.1 | PR #19912, design RFC #17799 |
| Docs page added | 2025-09-17 | PR #22919 |
| V0 per-request callable (`SamplingParams.logits_processors`) **removed** | v0.17.0 | PR #34400 |
| Current release (as of 2026-08-07) | v0.26.0 (2026-07-27) | pins torch==2.11.0, transformers>=5.5.3 |

**Constraints:**

- **Speculative decoding: hard-incompatible.** Engine-config validation raises `"Custom logits processors are not supported when speculative decoding is enabled."` Opt-in support = open unmerged PR #43672. You choose watermarking *or* spec-decode. `OFFICIAL-SRC`
- **Model Runner V2** doesn't support custom processors yet; silently falls back to the V1 runner (PR #47585 open, active 2026-08-04). Architectural churn is real — re-verify on every vLLM upgrade. `OFFICIAL-SRC`
- **Ordering:** structured-output grammar bitmasks are applied to logits *before* logits processors run (`STATIC` — read from `gpu_model_runner.py`/`sampler.py`). Mechanically composable with guided decoding, but constrained output has little entropy to watermark.
- **Tensor-parallel / prefix caching / chunked prefill:** no documentation or tests found either way. `OPEN` — must be answered by execution in Phase 1/5.

**Upstream watermarking activity: none.** No RFC, issue, PR, or discussion in vllm-project for content watermarking (exhaustive `gh` search across issues/PRs/code/discussions, 2026-08-07; `vllm-project/rfcs` is an empty repo). `OFFICIAL-SRC`

<a id="plugin-assessments"></a>
## 2. Plugin assessments (hands-on, 2026-08-07)

| | `eth-sri/unified-watermarking` | `dapurv5/vLLM-Watermark` | `THU-BPM/MarkLLM` | `transformers` built-ins + `google-deepmind/synthid-text` |
|---|---|---|---|---|
| Integration | Current V1 plugin API (`AdapterLogitsProcessor`); imports/signatures match v0.26.0 (`STATIC`) | Monkey-patches private `Sampler` via `llm.llm_engine.engine_core…model_runner.sampler`; needs `VLLM_ENABLE_V1_MULTIPROCESSING=0` (`STATIC`) | Research toolkit; has a vLLM demo script | HF `generate()` logits processors — not vLLM-wired, but the algorithm reference implementations |
| Server path (`vllm serve`) | Documented incl. `vllm_xargs` — **never executed by anyone we could verify** | **None.** Offline `LLM()` only | Demo-level | n/a |
| Detection | `detect()` per scheme (6 schemes); **`EXECUTED` on CPU** — KGW/Chi2/AAR p<1e-10 clean separation; SynthID after an init bug workaround (int vs float epsilon) | 9 detector variants (z-scores/p-values), `STATIC` | Per-algorithm | KGW `WatermarkDetector` **`EXECUTED`** (see §4); SynthID Bayesian detector requires training |
| Maturity | 1★, 2 commits, single author, last push 2026-02-11, no tests/CI, companion to arXiv 2602.06754 | 4★, author: "not production ready", no CI against real vLLM, latent `Sampler.forward()` signature drift | 1,000+★, Apache-2.0, actively maintained | Official, maintained, Apache-2.0 |
| License | **NO LICENSE FILE — do not copy code** (all-rights-reserved by default). Design reference only | Inconsistent: pyproject says MIT, LICENSE file + GitHub API say Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Verdict | Right architecture, research-grade; blocked for code reuse by missing license | Not viable for any serving deployment | Algorithm library — mine it, don't deploy it | **Primary implementation source** |

**Consequence for implementation:** write our own V1 `LogitsProcessor` from scratch, porting algorithm logic from Apache-2.0 sources only (`transformers`, `synthid-text`, MarkLLM). Use eth-sri solely as an existence proof of the wiring pattern (or obtain a license from the authors).

## 3. Watermarking science — what the implementation must respect

**Schemes that matter here:**

- **KGW / green-list** (Kirchenbauer et al.): keyed hash partitions vocab per step; bias δ added to "green" logits; detection = z-test on green fraction. Reference defaults: γ=0.25, δ=2.0, detect at z≥4.0 with `ignore_repeated_ngrams=True`. Official impl in `transformers`. `CORROBORATED`/`EXECUTED`
- **SynthID-Text** (Google DeepMind, Nature 634:818–823, 2024): tournament sampling; non-distortionary mode; detection via trained Bayesian detector (~10k matched watermarked/unwatermarked examples per generation config) or an untrained weighted-mean scorer. Open-sourced (Apache-2.0), in `transformers` ≥4.46. Only scheme with production evidence: deployed in Gemini; 20M-response live A/B showed thumbs-rate deltas of ±0.01–0.02pp (negligible quality impact for chat). `OFFICIAL-SRC`/`CORROBORATED`
- Gumbel/EXP (Aaronson), Unigram, DiPmark, SWEET etc.: implemented in MarkLLM; not first targets.

**Hard operational limits** (tell every stakeholder; the Code of Practice pre-accepts several of these):

1. **Entropy dependence.** Watermarks live in sampling randomness. Temperature 0 / greedy → little to no signal. Low-entropy outputs (code, JSON, function calling, structured output) are an unsolved research area (arXiv 2405.14604, 2506.06409). `CORROBORATED`
2. **Length dependence.** Reliable detection needs roughly 25–200+ tokens depending on scheme/settings; the Code's own threshold acknowledges this (sub-200-token free-form text excepted). `CORROBORATED`
3. **Paraphrase/translation strip the mark.** KGW: 99.8% → 9.7% TPR@1%FPR after 5 paraphrase rounds (arXiv 2303.11156). SynthID: F1 1.0 → 0.71 after one Chinese back-translation (arXiv 2508.20228). Retrieval/logging survives paraphrase (arXiv 2303.13408) — which is why logging is a recommended *supplementary* layer (and legally recognized in Recital 133, though insufficient alone). `CORROBORATED`
4. **Outlier study caution.** arXiv 2607.16010 (Jul 2026, small-sample preprint, wide CIs) reports pre-attack recall of only ~17–30% for KGW/Unigram/SynthID and 5.4% FPR for SynthID on human text — far worse than the established literature. Do not average it into claims; cite as an outlier if at all. `CORROBORATED` (that the paper says this), disputed (that it generalizes) |
5. **Key management.** Generator and detector share a secret. Self-hosting is an advantage (enterprise controls both ends) but key generation/storage/rotation/scoping has no off-the-shelf solution. Design needed. `OPEN`
6. **Multi-turn/agentic effects** (watermarked text re-entering context, tool outputs): open research gap; no quantified study found. `OPEN`
7. **Classifier detectors (GPTZero-style) are not marking.** They infer style, embed nothing, and are explicitly "not deemed mature enough" in the Code for compliance. Independent tests report double-digit FPRs. Not a compliance path. `CORROBORATED`

## 4. Executed local proof (2026-08-07)

CPU-only, transformers 4.57.6 / torch 2.9.1 / Python 3.14.4, gpt2, 60 new tokens, `WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")`, sampling at temperature 0.7. Full scripts + raw logs: [`../research/demo/`](../research/demo/).

| Input | z-score | p-value | Detector verdict |
|---|---|---|---|
| Watermarked generation | **5.37** | 5.4e-9 | AI-generated ✓ |
| Unwatermarked generation | 0.89 | 0.30 | not flagged ✓ |
| Human-written control | −0.54 | 0.58 | not flagged ✓ |

SynthID generation (`SynthIDTextWatermarkingConfig`) also confirmed working in the installed transformers version (detection not attempted — requires detector training). The observed ~48% CPU slowdown is a single-run, non-vLLM number — **not** a production overhead figure (gap D2).

## 5. Citations

- Kirchenbauer et al., "A Watermark for Large Language Models" — github.com/jwkirchenbauer/lm-watermarking
- Dathathri et al., "Scalable watermarking for identifying LLM outputs", Nature 634 (2024) — github.com/google-deepmind/synthid-text
- arXiv 2303.11156 (paraphrase attacks) · arXiv 2508.20228 (SynthID back-translation) · arXiv 2303.13408 (retrieval defense) · arXiv 2405.14604 / 2506.06409 (low-entropy) · arXiv 2607.16010 (forensic-readiness outlier) · arXiv 2405.10051 (MarkLLM) · arXiv 2602.06754 (unified framework, eth-sri)
- vLLM: docs.vllm.ai/en/latest/features/custom_logitsprocs/ · RFC #17799 · PRs #19912, #22919, #34400, #43672, #47585
