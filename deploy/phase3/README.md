# Phase 3 — detector service + standalone FMS Guardrails Orchestrator

Manifests in this directory stand up (1) the watermark detector service as a
built-on-cluster image (`detector-build.yaml`, `detector-deploy.yaml`) and
(2) a **standalone** (not RHOAI-managed) FMS Guardrails Orchestrator wired
to it in detection-only mode (`orchestrator.yaml`), on plain OpenShift 4.20
(`ocp-ai`) with **no RHOAI installed**. The RHOAI-managed Guardrails flavor
(operator-installed, dashboard-configured) is Phase 4
(`docs/implementation.md`) — this phase deploys the upstream
`foundation-model-stack/fms-guardrails-orchestrator` project directly.

The runbook was executed on the cluster on 2026-08-08. Every `oc`/`curl`/
`skopeo`/`gh api` command supporting the execution claims is recorded in
`EXPERIMENTS.md`; the manifests below remain the reproducible deployment
source.

CPU-only throughout (tokenizer + torch CPU scoring for the detector; a
stateless Rust HTTP proxy for the orchestrator) — **no GPU node needed**,
unlike `deploy/phase0`/`deploy/phase1`. Don't run `scripts/scale-gpu.sh 1`
for this phase.

## Prerequisites

```bash
export KUBECONFIG=cluster/auth/kubeconfig   # repo-root relative; gitignored, never print its contents
```

- **Namespace**: this phase reuses the `watermark` namespace created by
  `deploy/phase0/namespace.yaml` (`oc apply -f deploy/phase0/namespace.yaml`
  if starting from a clean cluster — idempotent either way, see that file's
  own comment).
- **Watermark key Secret**: this phase reuses the EXISTING `watermark-key`
  Secret from `deploy/phase0/secret-template.yaml` → `secret.yaml`
  (gitignored). If Phase 0/1 already ran on this cluster, it's already
  there; if not, follow `deploy/phase0/README.md` §1 to create it before
  continuing (`detector-deploy.yaml`'s env references it by name and will
  sit `CreateContainerConfigError` without it).
- **`pyyaml`**: used below only to validate this directory's own YAML syntax
  locally, not by anything that runs in-cluster. Already present on the
  local workstation (`python3 -c "import yaml; print(yaml.__version__)"` →
  `6.0.2` in the recorded environment — no `pip install` was needed).

## 1. Build the wheel

```bash
./deploy/phase0/build-wheel.sh
# -> dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl (re-run whenever
#    src/vllm_watermark/ changes; see that script's own header comment)
```

## 2. Build context safety — read before running `oc start-build`

`detector/Dockerfile` needs both `dist/*.whl` and `detector/` in its build
context, so the context can't just be the `detector/` directory alone. The
task brief's suggested command was `oc start-build detector --from-dir=.`
(repo root) — **this task deliberately does not recommend that literally**,
for a concrete reason: the repo root also contains `cluster/`, a **gitignored
directory holding live OpenShift credentials** (kubeconfig, kubeadmin
password, installer TLS material — see repo-root `.gitignore` and
`AGENTS.md` §3, "never commit, print, echo, or paste their contents").
`oc start-build --from-dir=DIR` archives and streams the literal contents of
`DIR` from the local filesystem (per OpenShift 4.20 docs,
`modules/builds-binary-source.adoc`, fetched from `openshift/openshift-docs`
branch `enterprise-4.20` since `docs.redhat.com` itself returned HTTP 403 to
every fetch attempt from this environment) — nothing in the fetched docs
confirms that step is filtered by `.dockerignore` the way a local `docker
build`'s context-collection step is. Rather than gamble live credentials on
an unconfirmed detail, stage a directory containing only what the Dockerfile
actually needs and point `--from-dir` at that instead:

```bash
rm -rf /tmp/detector-build-ctx
mkdir -p /tmp/detector-build-ctx/dist
cp -r detector /tmp/detector-build-ctx/detector
cp dist/vllm_watermark-*.whl /tmp/detector-build-ctx/dist/
# Sanity check: this must NOT print anything under cluster/, research/, aws, etc.
find /tmp/detector-build-ctx -type f
```

A repo-root `.dockerignore` (`cluster/`, `research/`, `benchmarks/data/`,
`benchmarks/results/`, `.git/`, `gpu/`, `*.pem`, `*.key`, `aws`) is also
provided as defense in depth for anyone who instead runs a plain local
`docker build` / `podman build -f detector/Dockerfile .` from the repo root
(that path DOES honor `.dockerignore`, per standard docker/podman/Buildah
semantics — see that file's own header comment) — it is not what the
`oc start-build` step below relies on for safety.

## 3. Create the ImageStream + BuildConfig, then build

```bash
oc apply -f deploy/phase3/detector-build.yaml
oc -n watermark start-build detector --from-dir=/tmp/detector-build-ctx --follow
```

Watch for `Push successful` at the end of the follow output, then confirm
the ImageStreamTag exists:

```bash
oc -n watermark get istag detector:latest
```

## 4. Deploy the detector service

```bash
oc apply -f deploy/phase3/detector-deploy.yaml
oc apply -f deploy/phase3/detector-synthid-deploy.yaml   # synthid-scheme twin; see that file's header
oc -n watermark wait --for=condition=Available deploy/detector-synthid --timeout=8m
oc -n watermark wait --for=condition=Available deploy/detector --timeout=5m
```

The Deployment's `image.openshift.io/triggers` annotation (see that file's
header comment for the exact OpenShift 4.20 doc citations) means the
placeholder `image:` field in the manifest gets overwritten with the real
built image automatically once step 3's build completes and updates the
`detector:latest` ImageStreamTag — `oc apply`-ing `detector-deploy.yaml`
*before* step 3 has ever produced a build is fine (the pod sits
`ImagePullBackOff` on the placeholder pull spec until the trigger fires),
but doing step 3 first, as ordered above, avoids that transient state.

Sanity-check the plugin loaded and the key is configured:

```bash
oc -n watermark port-forward svc/detector 8000:8000 &
curl -s http://localhost:8000/health
# -> {"status":"ok"}
curl -s http://localhost:8000/ready
# -> {"status":"ready","tokenizer_loaded":true,"key_ids":["<your key_id>"]}
#    (503 with tokenizer_loaded/keys_configured booleans if not yet ready —
#    see detector-deploy.yaml's readinessProbe comment for expected timing)
```

## 5. Deploy the orchestrator

```bash
oc apply -f deploy/phase3/orchestrator.yaml
oc -n watermark wait --for=condition=Available deploy/orchestrator --timeout=3m
```

```bash
oc -n watermark port-forward svc/orchestrator 8034:8034 &
curl -s http://localhost:8034/health
# -> {"fms-guardrails-orchestr8":"0.16.0"}
```

If this 503s or connection-refuses, check `oc -n watermark logs
deploy/orchestrator` first — a config parse error (e.g. a YAML typo in the
`orchestrator-config` ConfigMap) crash-loops the container immediately
rather than starting degraded, since the whole config is loaded once at
process start (`OrchestratorConfig::load`, `src/main.rs`).

## 6. Exercise both endpoints

Scheme routing is SERVER-SIDE: `watermark-kgw` routes to the kgw-scheme
`detector` Deployment and `watermark-synthid` to the dedicated
`detector-synthid` Deployment (each pins `WATERMARK_DETECTOR_SCHEME`), so
**both detector ids return correct verdicts with empty `detector_params`**
— verified live 2026-08-08, full matrix in `EXPERIMENTS.md`.
`detector_params.scheme` remains an optional per-request override (the
orchestrator forwards non-`threshold` params verbatim). Background: the
pinned orchestrator image predates `path_prefix` routing, which is exactly
why the twin-Deployment design exists — see `orchestrator.yaml`'s header.

### (a) Direct detector endpoint — `/v1/watermark/detect`

Port-forward from step 4 (`svc/detector` on `localhost:8000`) still needs to
be running.

```bash
# A text with genuinely high z-score is needed to see a `true` verdict —
# substitute a real Phase 1/2 watermarked sample from EXPERIMENTS.md /
# benchmarks/data/ output (gitignored, not checked in) rather than this
# placeholder, which is far too short to score meaningfully (see
# detector/app.py's `InsufficientTokensError` — KGW needs >= 2 tokens,
# SynthID needs >= ngram_len (default 5), or it 422s).
curl -s http://localhost:8000/v1/watermark/detect \
  -H 'Content-Type: application/json' \
  -d '{"text": "<paste a known-watermarked KGW sample here>", "scheme": "kgw"}'
```

Expected shape (fields per `detector/app.py::_build_detect_result`).
BOTH committed Deployments now set `SIGNING_KEY_PATH` from Secret
`detector-signing-key` (create it FIRST or the pod fails loudly at startup —
deliberate; see the manifest comment):

```bash
umask 077
openssl genpkey -algorithm ed25519 -out signing.pem      # keep OUTSIDE git
oc -n watermark create secret generic detector-signing-key \
  --from-file=signing.pem=signing.pem \
  --from-literal=SIGNING_KEY_ID=<your-key-id>            # id rotates WITH the key
```

Example below is the actually-captured response for a known-KGW sample
(EXPERIMENTS.md raw-evidence addendum, 2026-08-08), signature truncated:

```json
{
  "scheme": "kgw",
  "key_id": "<your key_id>",
  "verdict": true,
  "z_score": 12.817175976009691,
  "p_value": 6.569985692499835e-38,
  "score": 1.0,
  "num_tokens_scored": 400,
  "detector_version": "vllm-watermark-detector/0.1.0.dev0",
  "model_tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
  "scheme_details": {"num_green": 211, "gamma": 0.25},
  "signature": "eyJhbGciOiJFZERTQSIsImI2NCI6ZmFsc2UsImNy…",
  "signing": "enabled"
}
```

### (b) Orchestrator — `POST /api/v2/text/detection/content`

Port-forward `svc/orchestrator` on its main port too:
`oc -n watermark port-forward svc/orchestrator 8033:8033 &`

**KGW** (matches the task brief's literal example — empty `detector_params`
works here because the detector's `WATERMARK_DETECTOR_SCHEME` env default is
`kgw`, see detector-deploy.yaml):

```bash
curl -s http://localhost:8033/api/v2/text/detection/content \
  -H 'Content-Type: application/json' \
  -d '{
        "content": "<paste a known-watermarked KGW sample here>",
        "detectors": {"watermark-kgw": {}}
      }'
```

**SynthID** (empty params; the dedicated Service makes scheme authority
server-side):

```bash
curl -s http://localhost:8033/api/v2/text/detection/content \
  -H 'Content-Type: application/json' \
  -d '{
        "content": "<paste a known-watermarked SynthID sample here>",
        "detectors": {"watermark-synthid": {}}
      }'
```

Expected shape when a detection fires (`ContentAnalysisResponse`, per
`src/clients/detector.rs` at the pinned image's `0.17.0` vintage — field
names verified by reading that struct's `Deserialize` derive, not the task
brief's prose):

```json
{
  "detections": [
    {
      "start": 0,
      "end": 987,
      "text": "<the submitted content, echoed back>",
      "detection": "kgw-watermark",
      "detection_type": "watermark",
      "detector_id": "watermark-kgw",
      "score": 1.0,
      "metadata": {
        "z_score": 12.817175976009691,
        "p_value": 6.569985692499835e-38,
        "key_id": "<your key_id>",
        "scheme": "kgw",
        "num_tokens_scored": 254,
        "detector_version": "vllm-watermark-detector/0.1.0.dev0",
        "num_green": 210,
        "gamma": 0.25
      }
    }
  ]
}
```

Below the detector's own z-threshold (default z≥4.0), the detector service
returns an empty per-content list (deliberate "no detection" convention —
see `detector/app.py`'s docstring citation of the upstream
`detectors/huggingface/detector.py` behavior it mirrors), so the orchestrator
response for unwatermarked/human text is `{"detections": []}`, not an error.

`ContentAnalysisResponse.detector_id` disambiguation was also executed: one
request naming both detectors returned only the matching detection with the
correct detector id. The raw verdict matrix is in `EXPERIMENTS.md`.

## 7. Teardown

```bash
oc -n watermark delete -f deploy/phase3/orchestrator.yaml --ignore-not-found
oc -n watermark delete -f deploy/phase3/detector-synthid-deploy.yaml --ignore-not-found
oc -n watermark delete -f deploy/phase3/detector-deploy.yaml --ignore-not-found
oc -n watermark delete -f deploy/phase3/detector-build.yaml --ignore-not-found
rm -rf /tmp/detector-build-ctx
```

The `watermark` namespace and `watermark-key` Secret are left in place
deliberately (shared with Phase 0/1, cheap, no billing impact) — see
`deploy/phase0/README.md` §5 for the same convention. No GPU node was used
by this phase, so there is nothing to scale down.

## Acceptance evidence

The runbook above was executed end to end on the cluster on 2026-08-08 —
build, deploys, orchestrator wiring, and the full verdict matrix
(known-KGW and known-SynthID watermarked text detected by exactly their own
detector ids with empty params; clean and human text negative on both;
dual-detector requests attribute correctly). Raw transcripts and the exact
commands live in `EXPERIMENTS.md` (Phase 3 entry and the raw-evidence
addendum); this README carries no evidence of its own. Earlier revisions of
this section described the pre-execution state — see git history for that
provenance.

## YAML validation

```bash
python3 -c "
import pathlib, yaml
for p in sorted(pathlib.Path('deploy/phase3').glob('*.yaml')):
    docs = list(yaml.safe_load_all(p.read_text()))
    print(f'{p}: {len(docs)} document(s) OK')
"
```

Re-run the command above for the current result; it parses all five YAML
manifests. The repo-root `.dockerignore` is plain text and is not part of this
check. This confirms only that the files parse as valid YAML. The stronger
`oc apply --dry-run=server` check against the live OpenShift 4.20 API was
also executed for every Phase 3 object; its raw output and the bare vLLM
PodSecurity warning are recorded in `EXPERIMENTS.md`.
