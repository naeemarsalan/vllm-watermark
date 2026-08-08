# OpenShift AI integration facts

Verification tags per [`facts.md`](facts.md). Fetch dates: 2026-08-07/08.

## vLLM versions shipped (all recent releases clear the 0.10.1 plugin-API floor)

`OFFICIAL-SRC` — Red Hat "RHOAI and vLLM version compatibility" table, https://access.redhat.com/articles/rhoai-supported-configs-3.x

| RHOAI | vLLM shipped |
|---|---|
| 3.4 (current GA) | v0.18.0 (CUDA, ROCm, Power/Z, Spyre) · v0.17.1 (Gaudi) |
| 3.3 | v0.13.0 (CUDA/ROCm/Gaudi) · v0.10.1.1.6 (Power/Z) · v0.11.0–v0.12.0 (Spyre) |

Note: v0.18.0 is past the v0.17.0 removal of the old V0 per-request logits-processor API — any V0-era code is dead on RHOAI 3.4; only the V1 plugin API is relevant.

## Getting a plugin into the serving path

1. **Extra CLI args — documented and supported.** `OFFICIAL-SRC`
   - Dashboard "Additional serving runtime arguments" per deployment (→ `spec.predictor.model.args`), since RHOAI 2.16.
   - RHOAI 3.4 / llm-d: standard Kubernetes `args:` on `LLMInferenceService` (merged with defaults); legacy `VLLM_ADDITIONAL_ARGS` env var still works.
   - This is how `--logits-processors <FQCN>` reaches vLLM.
2. **Getting the plugin package into the image.** The plugin must be importable inside the runtime container. Documented route: duplicate the ServingRuntime and point it at a **custom image** (base: the Red Hat vLLM image + `pip install` of the plugin package; an entry point in group `vllm.logits_processors` makes loading automatic). `OFFICIAL-SRC` (procedure exists)
   - **Supportability caveat `OPEN`:** Red Hat's Container Support Policy (access.redhat.com/articles/2726611) does not support modified product images absent a product-specific carve-out; none found for RHOAI/RHAIIS. Needs an internal PM/support answer before production commitment. Alternatives to explore in Phase 4 include image-free injection (PYTHONPATH volume / init container); that route is undocumented, and its support posture is also unknown and requires confirmation.
3. **Per-request pass-through `OPEN` (C8):** whether KServe/Istio pass `vllm_xargs` in `extra_body` untouched — verify by execution in Phase 4.

## Detection side: guardrails

`OFFICIAL-SRC` — RHOAI 3.4 "Enabling AI safety with Guardrails" docs; github.com/trustyai-explainability/guardrails-detectors

- The FMS Guardrails Orchestrator accepts **any** detector server implementing the detectors API: `POST /api/v1/text/contents`.
- Shipped detectors today: built-in regex/file-type, Hugging Face classifier detectors (e.g. `granite-guardian-hap-38m`), NeMo Guardrails Presidio/regex rails. **No watermark detector exists.**
- **Lifecycle warning:** RHOAI 3.4 now labels FMS Guardrails **legacy**, says it will be deprecated in a future release, and directs users to NeMo Guardrails. `OFFICIAL-SRC` — [RHOAI 3.4 FMS Guardrails documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/enabling_ai_safety_with_guardrails/enabling-ai-safety-with-guardrails_safety), fetched 2026-08-08 (fact C11).
- The direct detector API and historical FMS detector-contract path were built and exercised on OpenShift; an upstream `nemoguardrails==0.23.0` custom action was also exercised (`EXECUTED`, fact D5 and [`EXPERIMENTS.md`](../EXPERIMENTS.md)). The RHOAI-managed `NemoGuardrails` custom-resource path remains `OPEN`; neither executed proof establishes a recommended or supported production architecture.
- The exact Code of Practice detection-access language is preserved verbatim in [`quotes.md`](quotes.md#guidelines-para-70) (`OJ-VERBATIM`). Apply it to a deployment only with the required legal and operational context; do not reduce it to an uncited general access rule.

## Models

Decode-time watermarking operates on the final logits tensor before sampling. Across the [Red Hat AI model families reviewed](https://huggingface.co/RedHatAI) (Llama, Granite, Mistral, Qwen, and DeepSeek), no model-specific API blocker was identified statically. `OFFICIAL-SRC` (mechanism + catalog) / `STATIC` (compatibility inference). Per-model execution, quality, and detectability remain `OPEN` for Phase 5, especially at low temperature or with structured output.

## Red Hat roadmap

Nothing public exists: no blog, doc, or release note mentions watermarking or AI Act text marking; public Jira search returned nothing, but RHOAIENG is not anonymously browsable, so that is a weak negative. Roadmap statements require an internal product-management conversation. `OPEN` (C6)
