# Agent kickoff prompt

The canonical prompt for an implementation agent picking up this repo. Paste it into a fresh agent session running on a machine with cluster access (`oc` logged in or `KUBECONFIG=cluster/auth/kubeconfig` available).

---

You are implementing EU AI Act text watermarking for vLLM on OpenShift AI in the repo `vllm-watermark`. This repo already contains a fully verified research base — do not re-litigate it, build on it.

Start by reading, in order: `AGENTS.md` (binding operating rules), `README.md`, `docs/facts.md`, `docs/implementation.md`. Skim `docs/technical.md` and `docs/openshift-ai.md` before writing any code, and `docs/cluster.md` before touching the cluster.

Your mission: execute the implementation plan in `docs/implementation.md`, phases 0 through 3, in order — (0) baseline vLLM serving on the `ocp-ai` cluster's GPU node, (1) a KGW watermark logits processor running end-to-end through `vllm serve` with statistical detection validated against watermarked/unwatermarked/human corpora, (2) SynthID-Text generation and detection using Google's open-sourced implementation, (3) the detector wrapped as a service implementing the TrustyAI Guardrails detector contract (`POST /api/v1/text/contents`) and validated through the GuardrailsOrchestrator on the cluster. Phase 4 (RHOAI ServingRuntime deployment) follows only after 1–3 pass their acceptance criteria.

Non-negotiable rules (full detail in AGENTS.md):
- Nothing "works" until you executed it and captured command + raw output in `EXPERIMENTS.md`. Static code reading is labeled STATIC, never presented as working.
- Every number is measured or cited — never recalled. Update `docs/facts.md` tags in the same commit as the evidence.
- Port algorithm code only from Apache-2.0 sources (`transformers`, `google-deepmind/synthid-text`, `MarkLLM`). Never copy from `eth-sri/unified-watermarking` — it has no license; design reference only.
- Secrets: the `aws` file and `cluster/` are gitignored live credentials — never commit or print them. Watermark keys come from env/Secrets, never hardcoded or logged.
- The GPU node is billable: scale it up with `./scripts/scale-gpu.sh 1` when starting, and **always scale to 0 before ending a session**.
- The local workstation cannot run vLLM (Python 3.14, no GPU) — all vLLM execution happens in pods on the cluster.
- Expected friction, verified in advance: custom logits processors error out with speculative decoding enabled (that's upstream, not your bug); Model Runner V2 falls back to V1; watermark signal degrades at temperature 0 and on structured output — measure and document these, don't fight them.
- No personal names, internal communications, or customer identifiers anywhere in this repo.

Definition of done for this engagement: Phase 1–3 acceptance criteria met with evidence in `EXPERIMENTS.md`, `docs/facts.md` gaps D1, D5, D8 closed (and D2/D3 if Phase 5 benchmarking is reached), and a reproducible `deploy/` path. Anything engineering cannot close (support-policy carve-out, legal grace-period scope) gets recorded as blocked, not improvised.
