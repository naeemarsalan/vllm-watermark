# Adversarial implementation review

Review date: 2026-08-08

Reviewed implementation: `76c196e` through `0cfedee`; post-review correction:
`124ca3f` (`STATIC`; repository history).

Scope: a historical review of the repository's 2026-08-08 implementation phases
0–3, execution evidence, detector service, OpenShift manifests, benchmark tooling,
and cluster closing state. It predates the current 2026-08-09 Phase 4/D10 rerun
(`STATIC`; this file's dated scope and the repository history above).

## Verdict

**Conditional pass for the engineering PoC; not production-ready and not an
RHOAI-complete deployment.**

Phases 0–3 each have executed acceptance evidence. KGW and SynthID-Text ran
through `vllm serve`; both were statistically detected against model and human
controls; the detector contract ran through the FMS Guardrails Orchestrator;
and an upstream NeMo Guardrails 0.23.0 custom action called the detector and
returned the expected block/pass decisions. These are `EXECUTED` results, not
static-code inferences. The central evidence is in [EXPERIMENTS.md](EXPERIMENTS.md).

The broader definition of done was only partially met at this review's timestamp.
D1 and D8 are closed. D5's executable FMS and upstream-NeMo portions were
closed, while the RHOAI-managed path and D9 remained open then. The subsequent
2026-08-09 run executed the current internal metadata-only broker/managed-NeMo
path and the bounded D10 one-in-`N` contract, and remediated/re-ran D9
(`EXECUTED`; [current Phase 4/D10 evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
External KServe/Istio pass-through, D6 supportability, D4 key lifecycle,
multi-replica/streaming behavior, and platform-wide retention remain `OPEN`
(same evidence; facts C4/C8/D4/D6/D10).

## Acceptance matrix

| Phase | Verdict | Evidence-based basis | Qualification |
|---|---|---|---|
| 0 — baseline serving | **PASS** (`EXECUTED`) | OpenAI-compatible vLLM 0.18.0 served the selected model on one A10G; 100/100 benchmark requests completed and baseline throughput/latency were recorded. | Bare OpenShift Pod, not an RHOAI ServingRuntime. |
| 1 — KGW | **PASS** (`EXECUTED`) | Corrected single-instance KGW ran through `vllm serve`; at 256-token truncation TPR was 1.000 on n=116, with FPR 0.000 on n=115 unwatermarked plus n=150 human controls. Per-request control, malformed-argument rejection, speculative-decoding rejection, temperature-0 behavior, structured-output composition, and overhead were exercised. | Structured-output z magnitude was measured in the superseded double-load window; true-delta-2 magnitude remains `OPEN`. |
| 2 — SynthID-Text | **PASS** (`EXECUTED`) | Reference-equivalence tests passed; GPU/CPU g-values matched in 20/20 trials; through-server detection measured TPR 1.000 at 200/256/512 tokens and FPR 0.000 on the scored controls. The six-row comparison scores each control with both detectors. | Quality work is an exploratory n=15 proxy, not a human evaluation. SynthID throughput was 287.82 output tok/s versus the 904.35 baseline in the recorded configuration. |
| 3 — detector + guardrails | **PASS for PoC** (`EXECUTED`) | `POST /api/v1/text/contents` ran through orchestrator 0.16.0 with correct KGW, SynthID, cross-scheme, clean, and human outcomes. Direct results were Ed25519-JWS signed and tamper rejection was executed. Upstream NeMo 0.23.0 blocked a known KGW sample, passed a human sample, and blocked the two preserved malformed-200 cases. | The detector-outage branch is `STATIC` for this historical upstream-PoC scope. The later current managed-NeMo path executed detector outage/fail-closed recovery and a finite hash-only scan; platform-wide retention remains `OPEN` (facts D5/D10; [current evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)). |
| 4 — RHOAI deployment | **NOT RUN in this 2026-08-08 review** (`OPEN` historical snapshot) | The later 2026-08-09 run recorded the RHOAI operator/ServingRuntime/InferenceService/internal predictor and metadata-only managed-NeMo broker path (`EXECUTED`; [current Phase 4/D10 evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)). | External KServe/Istio pass-through, supportability, and production readiness remain `OPEN`. |

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
| The NeMo action could fail open, trust malformed HTTP-200 responses, or tunnel detector-supplied strings into logs. | The committed code defaults closed (`STATIC`); malformed-200 cases and sanitized logging were `EXECUTED`. Adversarial marker leakage was 0 in the recorded 135-line capture. Live outage behavior for the historical upstream-PoC action remains unexecuted, while the current managed path's real detector-outage/fail-closed behavior is `EXECUTED` in bounded scope (facts D5/D10; [current evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)). |
| Phase 0/3 runbooks contradicted executed state, client routing, health version, teardown, and build source. | The runbooks now distinguish the recorded paths; teardown includes the SynthID Deployment, and the detector README points to the executed Dockerfile/build path. |
| A SynthID comment falsely said the public sampling-table seed was secret-derived. | Corrected: the table seed is recorded/defaulted separately; secret-derived layer keys provide keying. |

## Open findings and required actions

### 1. Detector numeric configuration was not fail-fast in this review — high

At the time of this 2026-08-08 review, `load_settings()` accepted `WATERMARK_Z_THRESHOLD=nan`, zero SynthID key
depth, zero SynthID n-gram length, and KGW gamma `2` (`EXECUTED`, fact B23).
Source inspection shows that readiness does not validate those fields, that
NaN comparison semantics can suppress positive verdicts, and that the other
values fail when their detector configurations are constructed (`STATIC`,
fact B23). No live invalid-configuration deployment was claimed in this
historical review; the subsequent bounded rerun rejected invalid startup
configuration, rebuilt the immutable detector image, and repeated the live
matrix (`EXECUTED`; [current Phase 4/D10 evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

Historical required action: validate type, range, and finiteness for every
numeric setting at startup; add negative tests; make readiness depend on valid
configuration; rebuild the detector image; and repeat the Phase 3 live verdict
matrix. That bounded D9 action is now executed; broader production hardening
remains outside this review (`EXECUTED` scoped / `OPEN`; [current Phase 4/D10
evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

### 2. RHOAI-managed guardrails path was unverified in this review — release blocker

The recorded Phase 3 stack was plain OpenShift plus a standalone FMS orchestrator.
The NeMo result uses the upstream Python package, not RHOAI's operator-managed
`NemoGuardrails` CR (`EXECUTED`/`OPEN`, fact D5).

Historical required action in Phase 4: install/use the target RHOAI version,
verify its shipped NeMo version and ConfigMap/action mounting, exercise the real
CR, and resolve the library's request-body/event-log retention behavior. The
first three integration checks and bounded hash-only scan now have scoped
execution evidence; platform-wide retention and supportability remain `OPEN`
([current Phase 4/D10 evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

### 3. PodSecurity hardening is incomplete — high for reusable deployment

Server-side dry-run warns that the bare vLLM Pod lacks explicit
`allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`,
`runAsNonRoot: true`, and a RuntimeDefault/Localhost seccomp profile
(`EXECUTED`). This did not prevent the spike from running under the recorded
cluster admission setup; reusable-deployment hardening remains `OPEN` for
Phase 4.

Required action: apply restricted-compatible security contexts in the Phase 4
runtime and validate them through the actual ServingRuntime/InferenceService.

### 4. Access control is deployment policy only — high if externally exposed

The detector has no application authentication or authorization (`STATIC`,
documented threat boundary). The committed manifest exposes a ClusterIP
Service (`STATIC`), but a future
Route/gateway could expose detection and key-id discovery if deployed without
network or identity controls.

Required action: define NetworkPolicy, service identity/mTLS, gateway
authentication, authorization, and detector-access policy before exposure.

### 5. Retention claims remain scoped — medium

The detector code is stateless, and finite log-window checks found none of the
tested distinctive sample substrings in the three service logs (`STATIC` +
`EXECUTED`, fact D5). This is not an
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

### 8. Immutable historical content remains a documentation boundary — low runtime risk

The active notes and runbooks listed in the reproducible artifact map no
longer contain the obsolete phase narration identified in the earlier review
(`STATIC`; current files and targeted Markdown scan).
Two binding exceptions remain: `EXPERIMENTS.md` is append-only and
`research/` is read-only. Those historical areas retain process prose, local
path text, and unsupported or superseded contextual claims; fact B24 limits
what may be relied upon. Whether historical evidence is exempt or requires a
separately authorized redaction remains `OPEN`. No repository-wide clean-
sweep claim is made.

The append-only log also records an unresolved identifier in published
history and the corresponding history-rewrite/credential-rotation disposition.
Those actions remain `OPEN` for explicit maintainer/security authorization
([recorded repository-policy disposition](EXPERIMENTS.md)); no identifier is
repeated here.

## Reproducible artifact map

- [EXPERIMENTS.md](EXPERIMENTS.md) — append-only commands and raw outputs.
- [docs/facts.md](docs/facts.md) — verification register and current gap status.
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

Keep the recorded Phase 0–3 state as an evidence-backed PoC milestone. Do not label it
an OpenShift AI production implementation or EU AI Act compliance solution.
Before a production candidate, preserve the scoped Phase 4/D10 and D9 evidence,
resolve the managed-NeMo/platform retention path, add restricted PodSecurity
and access controls, and complete the Phase 5 robustness/key-management work.
Legal grace-period
scope (D7) and support-policy posture (D6) remain decisions for counsel and
product/support owners, not engineering inference.
