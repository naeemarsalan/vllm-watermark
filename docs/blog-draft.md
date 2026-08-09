# Mark, detect, repeat: What we learned watermarking text with vLLM

*We built a custom plugin providing two text-watermarking schemes for vLLM, ran it on OpenShift, and measured what worked, what slowed down, and what is still open.*[^scope]

> **Before you try this:** This is a custom proof of concept, not a Red Hat product feature. We have not established product support for this deployment pattern, so do not treat it as a supported Red Hat method. We tested both the earlier bare-OpenShift path and the scoped internal Red Hat OpenShift AI ServingRuntime/managed-guardrails path; external KServe/Istio pass-through and production hardening remain open. It is engineering research, not legal advice or proof of compliance.[^scope]

The question looked simple: can we put a watermark on an LLM response?

If the response stays inside an application, adding a label or a metadata field is easy. The problem starts when somebody copies the text into an email, a document, or another chat. The visible badge and the JSON envelope stay behind. The words travel on their own.

That is where decode-time text watermarking gets interesting. Instead of attaching something to the response after generation, we make small, keyed changes to token selection while the model is generating. The output remains plain text, and a separate detector can look for the statistical signal later. Whether those changes affect quality is a separate question that we have not closed.[^mechanism][^limitations]

We wanted to know whether that could work through vLLM, what it would look like on OpenShift, and what evidence we would need before calling it more than an idea.

## Why this matters now

Article 113 says, “It shall apply from 2 August 2026.” Article 50(2) says providers “shall ensure that the outputs of the AI system are marked in a machine-readable format and detectable as artificially generated or manipulated.” Article 111(4) says providers of the named systems “placed on the market before 2 August 2026” must take the necessary Article 50(2) steps “by 2 December 2026.” Application of that text to an internal-only deployment remains `OPEN` for counsel.[^law]

The Commission's voluntary Code of Practice says, “For free-form text longer than 200 tokens, watermarking still needs to be applied.” The Commission's Guidelines say that marking “without the means for their detection being available” “will not suffice.” The 200-token sentence is Code text, not an Article 50 quotation.[^code]

So this is not just a generation problem. If we can add a signal but cannot find it reliably, operate the detector, or explain the result, we have only built half of the system.

## What does a text watermark actually do?

An LLM does not choose the next word directly. It produces a list of scores, called logits, for the tokens it could generate next.

A watermark processor changes those scores slightly before sampling. With KGW, for example, the previous token and a secret-derived key determine a preferred group of next tokens. The processor gives that group a small boost. One token tells us very little, but over a longer response the preferred tokens should appear more often than chance would predict.

The detector runs the same calculation in reverse. It needs the same tokenizer, algorithm settings, and key. It counts how often the generated tokens fall into the expected groups and returns a statistical score. A high score means the text is consistent with that watermark configuration. It does not prove that the text is true, safe, copyrighted, or written by a particular person. A negative result also does not prove that a human wrote it.[^mechanism]

We implemented two schemes:

- **KGW**, a relatively simple green-list watermark that is easy to inspect and test.
- **SynthID-Text**, Google's multi-depth tournament watermark, using the open Apache-2.0 implementation as the algorithm source.[^mechanism]

The model itself does not change. There is no fine-tuning and no checkpoint conversion. The change sits in the serving layer, after the model calculates logits and before vLLM samples the next token.[^architecture]

## So where does this live?

On the generation side, a small Python package plugs into vLLM's V1 decoding path. It contains one processor for KGW and another for SynthID. Once the wheel is installed, vLLM finds both through the package's entry points. The watermark key comes from an environment variable backed by an OpenShift Secret; it is never sent in the request.[^architecture][^request]

With the package installed and the watermark defaults and key references supplied by the deployment, the server starts normally:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

We set watermarking on by default for the serving deployment. A request can still choose the scheme and key ID through vLLM's `vllm_xargs` extension on its OpenAI-compatible endpoint:[^request]

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "prompt": "Explain why reproducible AI testing matters.",
  "max_tokens": 256,
  "temperature": 0.7,
  "vllm_xargs": {
    "watermark": "on",
    "watermark_scheme": "kgw",
    "watermark_key_id": "poc-2026-08"
  }
}
```

The key ID is an identifier, not the key. Malformed per-request watermark arguments, unknown per-request watermark fields, and unknown key IDs are rejected with an HTTP 400 instead of silently producing unwatermarked text.[^request]

There is one important loading detail: use the package entry points **or** the `--logits-processors` flag, not both.

We learned that the hard way.

## The result looked great. It was also wrong.

Our first KGW run produced very strong detector scores, but the performance cost looked worse than expected. The package had registered an entry point and the server command also passed the processor by its fully qualified class name. vLLM 0.18.0 loaded both copies. It did not deduplicate them, so KGW ran twice and roughly doubled the intended bias.[^double-load]

The debugging changed how we record a run. We now require the exact command and raw output before a number can move into the fact register. We marked the first measurements as superseded, removed the extra flag, regenerated the corpora, and ran the comparison again with one processor instance.

Here are the corrected results we are willing to stand behind.

## What we measured

The test used upstream vLLM 0.18.0, `Qwen/Qwen2.5-0.5B-Instruct`, one NVIDIA A10G, temperature 0.7, and separate watermarked, unwatermarked, and human-text corpora.

At a 200-token truncation, the results were:[^results]

| Detector | Watermarked text detected | Unwatermarked false positives | Human-text false positives |
|---|---:|---:|---:|
| KGW, delta 2.0 | 119/120 (99.2%) | 0/119 observed | 0/150 observed |
| SynthID, depth 30 | 120/120 (100%) | 0/119 observed | 0/150 observed |

At 256 tokens, KGW detected all 116 samples that were long enough to score at that length, and SynthID detected all 117. Each detector again produced zero observed false positives across 115 unwatermarked and 150 human samples.[^results]

Those are good proof-of-concept results. They are not universal error rates. They describe one small model, one set of configurations, one key, and these particular corpora. A production threshold needs broader content, languages, models, attacks, and a much larger false-positive study.

The runtime cost was also measurable:[^performance]

| Configuration | Aggregate output tokens/second | p50 latency |
|---|---:|---:|
| No plugin | 904.35 | 1.117 s |
| Both processors loaded, watermark off | 913.78 | 1.115 s |
| KGW on | 643.38 | 1.576 s |
| SynthID on | 287.82 | 3.563 s |

Under these matched benchmark runs, loading the processors but leaving watermarking off did not reduce the reported throughput or median latency. The baseline delivered 1.41 times KGW's throughput and 3.14 times SynthID's. That is useful for a proof of concept, but we would not stop there for a production service. Larger models, realistic batching, and optimization work still need to be measured.[^performance]

We also confirmed that custom logits processors and speculative decoding are incompatible in vLLM 0.18.0: the server rejects that combination at startup. It fails clearly, which is better than silently dropping the watermark, but it is still a design constraint.[^limitations]

## How we handled detection

We kept the detector as a separate service. It does not need a running vLLM engine, and generation and verification can be operated independently.[^detection]

We deployed KGW and SynthID detector instances on OpenShift and routed requests through the FMS Guardrails Orchestrator. Known watermarked samples went to the correct detector. Cross-scheme checks, unwatermarked model output, and human text returned no detection. The direct endpoint also produced an Ed25519-signed result; the signature verified, and a tampered payload failed verification.[^detection]

The FMS test established the recorded success path through that standalone orchestrator. But Red Hat OpenShift AI 3.4 now labels FMS Guardrails as legacy and points users toward NeMo Guardrails, so this result is not evidence for a new long-term RHOAI design.[^support]

For the newer direction, we first called the same detector from a custom rail in the upstream NeMo Guardrails library. On that historical request path, the rail blocked a known KGW sample, passed a human sample, and failed closed on the two malformed-response cases preserved in the run log.[^detection]

We then executed the current RHOAI-managed `NemoGuardrails` path. A synchronous validation gateway selected every `N`th completed response, sent bounded correlation metadata through the managed action to an authenticated broker, and kept the exact pending response inside the gateway for detector authority. The fixed runs selected 20/20 responses at `N=1` and 20/100 at `N=5`; both KGW and SynthID positive cases mapped to the managed block action, clean controls passed, and a real detector outage exhausted three attempts before a content-free fail-closed response.[^continuous]

A later negative probe found another production gap: `load_settings()` accepted four invalid numeric configurations, including a `NaN` threshold (`EXECUTED`). The first remediation rejected those values but still lacked several upper bounds; a later review also found an explicit-blank vocabulary bypass. The current detector validates blank, lower, upper, and non-finite settings through lifespan; its immutable image matched local source, passed every built-image blank/maximum/overflow probe, failed a controlled blank-valued rollout before readiness, recovered, and then completed the full generated-response D10 matrix (`EXECUTED`). Request/cache limits and generation-side bound parity remain open.[^detector-config]

## Where Red Hat helps, and where it does not

RHOAI 3.4 includes a vLLM version with the V1 custom logits-processor interface. OpenShift let us store the key in a Secret, pin the serving image, roll out generation and detection separately, check service health, and rerun the tests in a controlled environment.[^red-hat]

That does not make this plugin a supported Red Hat feature. The watermark code, detector, validation gateway, manifests, and benchmark harness are custom. The internal RHOAI ServingRuntime/predictor and managed-guardrails path executed, but the external KServe/Istio pass-through remains untested. Red Hat's own EU AI Act page also draws a useful boundary: the customer's AI application may be classified under the Act; the underlying platform is not represented as “EU AI Act certified.”[^support]

That is the useful boundary. The recorded OpenShift environment hosted and repeated these controls. The platform result does not decide legal scope, choose a detection threshold, or turn experimental code into a supported compliance feature.[^scope]

## How we keep checking it

The watermark is tied to more than the source code. A model change, tokenizer update, new vLLM image, different sampling settings, key rotation, or serving configuration can change the result.

Our validation loop is straightforward:

1. **Pin the environment.** Record the model, tokenizer, image digest, vLLM version, algorithm settings, detector version, and key ID. Never record the secret.
2. **Generate matched samples.** Use the same prompts for watermarked and unwatermarked output, then add separately sourced human text.
3. **Run detection on everything.** Report the complete score distributions and false positives, not just the cleanest examples.
4. **Try the awkward cases.** Short responses, temperature zero, structured output, paraphrasing, translation, and incompatible serving options belong in the test plan.
5. **Measure the cost.** Compare throughput, latency, and quality against the same pinned baseline.
6. **Keep the evidence and repeat.** Preserve the command and raw output, then rerun after every model, tokenizer, processor, key-policy, or serving-stack change.[^verification]

The double-loading bug is the best argument for this loop. The code ran. The detector fired. The first chart would have looked impressive. The result was still misleading until we checked how the processor was loaded and repeated the test.

## Where we are now

We have a working proof of concept: two watermark schemes generating through vLLM, independent detection, per-request controls, measured performance, and configurable one-in-`N` validation through the scoped internal RHOAI managed-guardrails path.[^scope][^continuous]

We do not yet have a production solution. External gateway pass-through, caller authentication/network policy, multi-replica and streaming behavior, HA, and platform-wide retention remain untested. Key generation, rotation, application scoping, and compromise handling need a proper design. Quality needs a real evaluation. Paraphrasing and translation need robustness tests. Tensor parallelism and larger models need their own runs.[^limitations][^support]

So, can vLLM carry a machine-detectable text watermark? Under the recorded model and configuration, yes. Did the scoped internal RHOAI serving and managed-validation path execute? Yes. Does that prove an externally exposed, supported, production-ready design? No.[^scope][^support]

That is where we stop the claim. Calling the system compliant or supported would require the full technical evidence, the actual deployment context, Red Hat support guidance, and counsel.[^support]

The implementation, manifests, tests, and append-only experiment record are available in this repository: start with the [implementation plan](implementation.md), [fact register](facts.md), and [executed experiment log](../EXPERIMENTS.md).

---

## Sources and verification notes

[^scope]: `EXECUTED` (bare OpenShift and scoped internal RHOAI paths) / `OPEN` (external gateway and supportability) — [facts C4, C8, C9, C10, D1, D5, D6, and D10](facts.md), [current execution record](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted), and [Red Hat's Container Support Policy](https://access.redhat.com/articles/2726611).
[^law]: `OJ-VERBATIM` / `OPEN` (internal-only transition applicability) — [facts A1–A4 and verified quotations](quotes.md), [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689), and [Regulation (EU) 2026/1744](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32026R1744).
[^code]: `OJ-VERBATIM` — [facts A7, A9, and A10](facts.md), the Commission's [Code of Practice on Transparency of AI-Generated Content](https://ec.europa.eu/newsroom/dae/redirection/document/129555), and the [Article 50 Guidelines](https://ec.europa.eu/newsroom/dae/redirection/document/131215).
[^mechanism]: `OFFICIAL-SRC` / `STATIC` / `EXECUTED` — [facts B13, B15, D1, and D8](facts.md), the [technical explanation](technical.md), and the recorded [Phase 1/2 serving evidence](../EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8). The historical CPU report is `OPEN` under B14.
[^architecture]: `OFFICIAL-SRC` / `STATIC` — [facts B3–B4](facts.md), [vLLM custom logits-processor documentation](https://docs.vllm.ai/en/latest/features/custom_logitsprocs/), and the implemented [KGW](../src/vllm_watermark/kgw/processor.py) and [SynthID](../src/vllm_watermark/synthid/processor.py) processors.
[^request]: `EXECUTED` — [fact D1](facts.md) and the [Phase 1 per-request validation record](../EXPERIMENTS.md#per-request-control--validation-executed).
[^double-load]: `EXECUTED` — [vLLM API note and independent reproduction](api-notes-vllm-v0.18.0.md#10-plugin-loading-has-no-deduplication--entry-points--fqcn-flag-double-load), [Phase 1 correction](../EXPERIMENTS.md#2026-08-08--correction-phase-1-ran-two-kgw-processor-instances-effective-delta-40), and [fact D1](facts.md).
[^results]: `EXECUTED` — [scheme-comparison v2](../EXPERIMENTS.md#2026-08-08--scheme-comparison-v2-per-scheme-control-fpr-supersedes-the-v1-tables-control-rows), [facts D1 and D8](facts.md).
[^performance]: `EXECUTED` — [fact D2](facts.md) and the corrected [Phase 1/2 performance record](../EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8).
[^limitations]: `EXECUTED` / `CORROBORATED` / `OPEN` — [facts B7, B10, B17, B18, B23, D2–D4, and D9](facts.md).
[^detection]: `EXECUTED` — [facts B23, D5, and D9](facts.md), the [Phase 3 experiment](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half), and the [current managed-path evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted).
[^continuous]: `EXECUTED` (single-replica synchronous non-streaming PoC scope) / `OPEN` (production boundaries) — [fact D10](facts.md), the [acceptance contract](implementation.md#continuous-validation-acceptance), and the [current execution record](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted).
[^detector-config]: `EXECUTED` / `OPEN` (negative probe, rebuilds, current blank/bound/startup probes and matrix rerun, and remaining hardening) — [facts B23, D9, and D10](facts.md), the [original negative probe](../EXPERIMENTS.md#2026-08-08--independent-post-push-review-correction), the [current detector reconciliation](../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09), and the [current build-5 D10 rerun](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09).
[^red-hat]: `OFFICIAL-SRC` / `STATIC` / `EXECUTED` — [facts C1–C3, C8–C10, D1, D5, and D10](facts.md), the [current execution record](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted), [RHOAI supported configurations](https://access.redhat.com/articles/rhoai-supported-configs-3.x), and [OpenShift AI model-serving configuration](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html-single/configuring_your_model-serving_platform/configuring_your_model-serving_platform).
[^support]: `OFFICIAL-SRC` / `OPEN` — [facts C4, C8, C10, C11, and D5–D6](facts.md), [Red Hat's EU AI Act page](https://access.redhat.com/compliance/eu-ai-act), and [RHOAI 3.4 Guardrails documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/enabling_ai_safety_with_guardrails/enabling-ai-safety-with-guardrails_safety).
[^verification]: `STATIC` (method) / `EXECUTED` (recorded runs and corrections) — [implementation phases](implementation.md), [fact register](facts.md), and [append-only experiment record](../EXPERIMENTS.md).
