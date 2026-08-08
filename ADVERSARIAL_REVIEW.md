# Adversarial implementation review

Review date: 2026-08-08

Reviewed implementation: `76c196e` through `0cfedee`

Post-review correction: `124ca3f`
Scope: implementation phases 0–3, execution evidence, detector service,
OpenShift manifests, benchmark tooling, and cluster closing state.

## Verdict

**Conditional pass for the engineering PoC; not production-ready and not an
RHOAI-complete deployment.**

Phases 0–3 each have executed acceptance evidence. KGW and SynthID-Text ran
through `vllm serve`; both were statistically detected against model and human
controls; the detector contract ran through the FMS Guardrails Orchestrator;
and an upstream NeMo Guardrails 0.23.0 custom action called the detector and
returned the expected block/pass decisions. These are `EXECUTED` results, not
static-code inferences. The central evidence is in [EXPERIMENTS.md](EXPERIMENTS.md).

The broader definition of done is only partially met. D1 and D8 are closed.
D5's executable FMS and upstream-NeMo portions are closed, but the
RHOAI-managed `NemoGuardrails` CR path, shipped version, and retention behavior
remain `OPEN`. Phase 4 has not been performed. A new high-priority detector
configuration defect, D9, was also demonstrated during this review.

## Acceptance matrix

| Phase | Verdict | Evidence-based basis | Qualification |
|---|---|---|---|
| 0 — baseline serving | **PASS** (`EXECUTED`) | OpenAI-compatible vLLM 0.18.0 served the selected model on one A10G; 100/100 benchmark requests completed and baseline throughput/latency were recorded. | Bare OpenShift Pod, not an RHOAI ServingRuntime. |
| 1 — KGW | **PASS** (`EXECUTED`) | Corrected single-instance KGW ran through `vllm serve`; at 256-token truncation TPR was 1.000 on n=116, with FPR 0.000 on n=115 unwatermarked plus n=150 human controls. Per-request control, malformed-argument rejection, speculative-decoding rejection, temperature-0 behavior, structured-output composition, and overhead were exercised. | Structured-output z magnitude was measured in the superseded double-load window; true-delta-2 magnitude remains `OPEN`. |
| 2 — SynthID-Text | **PASS** (`EXECUTED`) | Reference-equivalence tests passed; GPU/CPU g-values matched in 20/20 trials; through-server detection measured TPR 1.000 at 200/256/512 tokens and FPR 0.000 on the scored controls. The six-row comparison scores each control with both detectors. | Quality work is an exploratory n=15 proxy, not a human evaluation. SynthID throughput was 287.82 output tok/s versus the 904.35 baseline in the recorded configuration. |
| 3 — detector + guardrails | **PASS for PoC** (`EXECUTED`) | `POST /api/v1/text/contents` ran through orchestrator 0.16.0 with correct KGW, SynthID, cross-scheme, clean, and human outcomes. Direct results were Ed25519-JWS signed and tamper rejection was executed. Upstream NeMo 0.23.0 blocked a known KGW sample and passed a human sample; malformed responses and detector outage failed closed. | FMS packaging is legacy in RHOAI 3.4. The managed NeMo CR/operator path remains `OPEN`; default NeMo 422/event logging conflicts with the desired retention posture. |
| 4 — RHOAI deployment | **NOT RUN** (`OPEN`) | No RHOAI operator, ServingRuntime, InferenceService, or managed `NemoGuardrails` CR was installed. | Required before claiming an OpenShift AI production deployment. |

All figures above are taken from the raw-command record and the registered
facts: [Phase 0–3 evidence](EXPERIMENTS.md), [facts B23 and D1–D9](docs/facts.md).

## Verification performed

The implementation history records these executed checks:

- Full combined test suite at the published audit revision: **154 passed** in
  1746.78 seconds (`EXECUTED`).
- Reference equivalence: **6 KGW tests passed** and **19 SynthID tests passed**
  (`EXECUTED`).
- Post-review focused service/plugin suite: **129 passed** in 125.36 seconds
  (`EXECUTED`).
- All five Phase 3 YAML files parsed locally, and all Phase 0/3 objects were
  accepted by `oc apply --dry-run=server` (`EXECUTED`).
- Live close-out: detector, detector-synthid, and orchestrator Deployments were
  each 1/1; orchestrator reported both detector services healthy; the GPU
  MachineSet desired/current replica counts were 0/0 (`EXECUTED`).
- Staged-diff secret scans found no private-key block, AWS access-key pattern,
  or watermark-secret value (`EXECUTED`, pattern-limited scan—not a formal
  secret-scanning guarantee).

Commands and raw outputs for the post-review checks are in the
[independent post-push review correction](EXPERIMENTS.md#2026-08-08--independent-post-push-review-correction).

## Resolved findings

The following defects were found during adversarial review and corrected:

| Finding | Resolution |
|---|---|
| KGW was initially loaded twice through both entry points and an explicit FQCN flag, doubling the effective bias. | The manifest now uses entry points only; all primary KGW numbers were re-measured with one instance. |
| SynthID orchestrator routing could fall back to KGW when client scheme parameters were omitted. | Separate KGW and SynthID Services/Deployments now provide server-side scheme authority; empty-parameter routing passed live. |
| Both detector Services initially selected the same pods. | Distinct component labels and disjoint endpoints were applied and verified. |
| The SynthID image-change trigger named the wrong container. | Its trigger field path now names `detector-synthid`. |
| Signing configuration allowed a deployed key without a stable key id. | Both detector manifests require `SIGNING_KEY_ID` from the signing Secret. |
| Comparison controls were not independently scored by both detectors. | The six-row comparison now reports KGW and SynthID FPR separately at every available length. |
| The two pytest trees collided during combined collection. | Importlib collection and canonical imports produced one reproducible 154-test command. |
| The NeMo action could fail open, trust malformed HTTP-200 responses, or tunnel detector-supplied strings into logs. | Default failure is closed; verdict type is checked; log output uses request-local identifiers and numeric/type-safe fields. Adversarial marker leakage was 0 in the recorded 135-line capture. |
| Phase 0/3 runbooks contradicted executed state, client routing, health version, teardown, and build source. | Corrected in `124ca3f`; teardown now includes the SynthID Deployment and the detector README points to the executed Dockerfile/build path. |
| A SynthID comment falsely said the public sampling-table seed was secret-derived. | Corrected: the table seed is recorded/defaulted separately; secret-derived layer keys provide keying. |

## Open findings and required actions

### 1. Detector numeric configuration is not fail-fast — high

`load_settings()` accepts `WATERMARK_Z_THRESHOLD=nan`, zero SynthID key
depth, zero SynthID n-gram length, and KGW gamma `2` (`EXECUTED`, fact B23).
A NaN threshold can silently suppress every positive verdict while the service
still appears ready; the other values fail later on request paths.

Required action: validate type, range, and finiteness for every numeric setting
at startup; add negative tests; make readiness depend on valid configuration;
rebuild the detector image; and repeat the Phase 3 live verdict matrix. This is
registered as D9.

### 2. RHOAI-managed guardrails path is unverified — release blocker

The current live stack is plain OpenShift plus a standalone FMS orchestrator.
The NeMo result uses the upstream Python package, not RHOAI's operator-managed
`NemoGuardrails` CR (`EXECUTED`/`OPEN`, fact D5).

Required action in Phase 4: install/use the target RHOAI version, verify its
shipped NeMo version and ConfigMap/action mounting, exercise the real CR, and
resolve the library's request-body/event-log retention behavior.

### 3. PodSecurity hardening is incomplete — high for reusable deployment

Server-side dry-run warns that the bare vLLM Pod lacks explicit
`allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`,
`runAsNonRoot: true`, and a RuntimeDefault/Localhost seccomp profile
(`EXECUTED`). This did not prevent the spike from running under the cluster's
admission setup, but it is not an acceptable reusable production posture.

Required action: apply restricted-compatible security contexts in the Phase 4
runtime and validate them through the actual ServingRuntime/InferenceService.

### 4. Access control is deployment policy only — high if externally exposed

The detector has no application authentication or authorization (`STATIC`,
documented threat boundary). It is currently a ClusterIP service, but a future
Route/gateway could expose detection and key-id discovery if deployed without
network or identity controls.

Required action: define NetworkPolicy, service identity/mTLS, gateway
authentication, authorization, and detector-access policy before exposure.

### 5. Retention claims remain scoped — medium

The detector code is stateless and log-window substring checks found no sample
plaintext in the three service logs (`STATIC` + `EXECUTED`). This is not an
absolute zero-retention proof. The FMS response contract echoes detected text,
and NeMo 0.23.0 defaults can echo request bodies in 422 responses and store full
message content in its event log.

Required action: write a data-flow/retention threat model, disable or replace
content-bearing NeMo paths, test proxy/platform logs, and verify the deployed
configuration over a defined retention interval.

### 6. Reproducibility is version-resolved, not supply-chain complete — medium

The NeMo constraints pin 78 resolved Python package versions, but the test base
image is mutable and hashes are absent. The detector Dockerfile also uses a
mutable Python base tag and pins direct dependencies without a fully hashed
transitive lock (`STATIC`; limitations are documented).

Required action: pin base images by digest, produce hashed lock files/SBOMs,
record the built detector digest in a release manifest, and add provenance and
vulnerability scanning.

### 7. Quality and robustness evidence is PoC-grade — medium

The recorded quality comparison is a small, single-model perplexity/diversity
proxy; it is not human-rated. Paraphrase, translation, code-heavy output,
stricter structured schemas, larger models, realistic batches, tensor
parallelism, prefix caching, and key rotation are not closed (`OPEN`, Phase 5).

Required action: execute Phase 5 with pre-registered corpora, paired statistics,
confidence intervals, human evaluation, adversarial transformations, and
operational key-management tests.

### 8. Repository process/content policy is not fully clean — low runtime risk

The append-only evidence honestly records phase-ordering and same-commit
discipline violations. It also contains historical workflow narration, while
several research/API-note files retain internal task labels (`STATIC`). The
earlier unqualified phrase-sweep claim was corrected in the append-only log.

Required action: perform a maintainer-approved content scrub of non-evidence
files and define whether immutable historical evidence is exempt or must be
rewritten/redacted. Do not claim this policy is closed until an explicit scan
scope passes.

### 9. Published-history hygiene requires a maintainer decision — low runtime risk

Earlier published commits contain attribution trailers and a pre-redaction
sandbox-domain blob, as already recorded in `EXPERIMENTS.md`. Removing them
would require a history rewrite and force-push (`OPEN` governance decision).
Rotate any credential associated with the historical sandbox domain rather
than treating a later textual redaction as credential invalidation.

## Reproducible artifact map

- [EXPERIMENTS.md](EXPERIMENTS.md) — append-only commands and raw outputs.
- [docs/facts.md](docs/facts.md) — verification register and D1–D9 status.
- [deploy/phase0/README.md](deploy/phase0/README.md) — executed bare-pod vLLM
  runbook and mandatory GPU scale-down.
- [deploy/phase3/README.md](deploy/phase3/README.md) — detector/orchestrator
  build, deploy, exercise, and teardown path.
- [detector/app.py](detector/app.py) — TrustyAI contract and direct detector
  service.
- [deploy/phase3/nemo-guardrails-poc.yaml](deploy/phase3/nemo-guardrails-poc.yaml)
  — hardened upstream-NeMo custom action PoC.
- [benchmarks/compare_schemes.py](benchmarks/compare_schemes.py) — six-row
  KGW/SynthID comparison.
- [benchmarks/quality_spotcheck.py](benchmarks/quality_spotcheck.py) — scoped
  exploratory quality proxy.

## Release recommendation

Keep `0cfedee`/`124ca3f` as an evidence-backed PoC milestone. Do not label it
an OpenShift AI production implementation or EU AI Act compliance solution.
Before a production candidate, close D9, execute Phase 4 on RHOAI, resolve the
managed-NeMo retention path, add restricted PodSecurity and access controls,
and complete the Phase 5 robustness/key-management work. Legal grace-period
scope (D7) and support-policy posture (D6) remain decisions for counsel and
product/support owners, not engineering inference.
