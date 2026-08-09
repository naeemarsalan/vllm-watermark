# OpenShift AI integration facts

Verification tags per [`facts.md`](facts.md). Fetch dates: 2026-08-07/08.

## vLLM versions shipped (all recent releases clear the 0.10.1 plugin-API floor)

`OFFICIAL-SRC` — Red Hat "RHOAI and vLLM version compatibility" table, https://access.redhat.com/articles/rhoai-supported-configs-3.x

| RHOAI | vLLM shipped |
|---|---|
| 3.4 (current GA) | v0.18.0 (CUDA, ROCm, Power/Z, Spyre) · v0.17.1 (Gaudi) |
| 3.3 | v0.13.0 (CUDA/ROCm/Gaudi) · v0.10.1.1.6 (Power/Z) · v0.11.0–v0.12.0 (Spyre) |

The RHOAI 3.4 builds listed here (v0.17.1–v0.18.0) are at or beyond the
v0.17.0 removal of the old V0 per-request logits-processor API; the
documented V1 plugin API is the applicable extension surface
(`OFFICIAL-SRC`, facts B3/B5).

## Getting a plugin into the serving path

1. **Extra CLI args — documented configuration surface.** `OFFICIAL-SRC`
   - Dashboard "Additional serving runtime arguments" per deployment (→ `spec.predictor.model.args`), since RHOAI 2.16.
   - RHOAI 3.4 / llm-d: standard Kubernetes `args:` on `LLMInferenceService` (merged with defaults); legacy `VLLM_ADDITIONAL_ARGS` env var still works.
   - A runtime can pass `--logits-processors <FQCN>` to vLLM, but this repository's installed wheel already registers both processors as entry points. The corrected executed path used entry points only; combining an entry point with the same FQCN flag double-loaded KGW (`EXECUTED`; [vLLM API note](api-notes-vllm-v0.18.0.md#10-plugin-loading-has-no-deduplication--entry-points--fqcn-flag-double-load)).
2. **Getting the plugin package into the image.** The plugin must be importable inside the runtime container. Duplicating a ServingRuntime and pointing it at a **custom image** is the documented Phase 4 pattern (`OFFICIAL-SRC`, fact C3); the internal RHOAI ServingRuntime/InferenceService predictor path executed with the package installed in the pinned image (`EXECUTED`; [current Phase 4/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)). The package's `vllm.logits_processors` entry points load the processors automatically (`STATIC`, fact B4).
   - **Supportability caveat `OPEN`:** Red Hat's [Container Support Policy](https://access.redhat.com/articles/2726611) addresses modified product images, but the RHOAI-specific answer requires product-management/support confirmation (facts C4/D6). Image-free injection through a `PYTHONPATH` volume or init container is another unexecuted, undocumented candidate; its support posture is also `OPEN`.
3. **Per-request pass-through `OPEN` (C8):** the internal predictor accepted the recorded request controls, but external KServe/Istio preservation of `vllm_xargs` in `extra_body` remains unexecuted (`OPEN`; [current Phase 4/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

## Detection side: guardrails

`OFFICIAL-SRC` — RHOAI 3.4 "Enabling AI safety with Guardrails" docs; github.com/trustyai-explainability/guardrails-detectors

- The FMS Guardrails Orchestrator documents the detector API `POST /api/v1/text/contents` (`OFFICIAL-SRC`, fact C5); the repository service executed that success path (`EXECUTED`, D5).
- The reviewed RHOAI 3.4 lists named regex/file-type, classifier, and Presidio/regex detectors. No watermark detector was identified in those dated sources (`OFFICIAL-SRC` for the lists / `OPEN` for the search-bounded absence, fact C5).
- **Lifecycle warning:** RHOAI 3.4 now labels FMS Guardrails **legacy**, says it will be deprecated in a future release, and directs users to NeMo Guardrails. `OFFICIAL-SRC` — [RHOAI 3.4 FMS Guardrails documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/enabling_ai_safety_with_guardrails/enabling-ai-safety-with-guardrails_safety), fetched 2026-08-08 (fact C11).
- The direct detector API and historical FMS detector-contract path were built and exercised on OpenShift; an upstream `nemoguardrails==0.23.0` custom action was also exercised (`EXECUTED`, fact D5). The current RHOAI-managed `NemoGuardrails` resource and metadata-only broker path then executed in the bounded one-replica synchronous validation run (`EXECUTED`; [current Phase 4/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)). External KServe/Istio pass-through, supportability, and platform-wide retention remain `OPEN` (same evidence; facts C8/D6/D10).
- The B23 probe showed an early revision accepting four invalid/non-finite values (`EXECUTED`), and later reviews found missing upper bounds and an explicit-blank vocabulary bypass after the first remediation. The current immutable image matched local source, rejected both blank forms, passed 9/9 built-image maximum/overflow pairs, failed a real blank-valued rollout before readiness, recovered, and completed the full D10 fixed matrix (`EXECUTED`; [current detector reconciliation](../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09); [current build-5 D10 rerun](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)). Request/cache limits and generation-side bound parity remain `OPEN` (fact D9).
- The exact Code of Practice detection-access language is preserved verbatim in [`quotes.md`](quotes.md#guidelines-para-70) (`OJ-VERBATIM`). Apply it to a deployment only with the required legal and operational context; do not reduce it to an uncited general access rule.

## Models

Decode-time watermarking operates on the final logits tensor before sampling. Across the [Red Hat AI model families reviewed](https://huggingface.co/RedHatAI) (Llama, Granite, Mistral, Qwen, and DeepSeek), no model-specific API blocker was identified statically. `OFFICIAL-SRC` (mechanism + catalog) / `STATIC` (compatibility inference). Per-model execution, quality, and detectability remain `OPEN` for Phase 5, especially at low temperature or with structured output.

## Red Hat roadmap

The searches recorded on 2026-08-07 found no public Red Hat blog, documentation, release note, or anonymously searchable issue mentioning text watermarking or AI Act text marking. This is a weak, dated negative and does not establish a roadmap position. `OPEN` (fact C6; source: searches registered in [`facts.md`](facts.md)).
