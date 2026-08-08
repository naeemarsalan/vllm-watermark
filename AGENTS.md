# AGENTS.md — operating rules for this repository

Mission: implement and validate decode-time text watermarking (generation + detection) for vLLM on OpenShift AI, per [`docs/implementation.md`](docs/implementation.md). The regulatory *why* is in [`README.md`](README.md) and [`docs/facts.md`](docs/facts.md).

## 1. Verification discipline (non-negotiable)

- **No assumptions presented as facts.** Every claim you add to docs must carry a tag from [`docs/facts.md`](docs/facts.md) (`OJ-VERBATIM` / `OFFICIAL-SRC` / `EXECUTED` / `STATIC` / `CORROBORATED` / `OPEN`) and a source.
- **"Works" means EXECUTED.** Never claim something runs until you ran it and captured the command + raw output. Static code reading earns `STATIC`, nothing more. The central lesson of the research phase: both prior-art plugins *looked* plausible; only execution separates them.
- **Numbers come from measurements or citations — never from memory.** If you quote a paper's number, link the paper. If you measure, record environment, command, and raw output in `EXPERIMENTS.md` (create at repo root on first run; append-only; one dated section per run).
- **Update `docs/facts.md` in the same commit** as the evidence that upgrades/invalidates a fact (e.g., D1 → EXECUTED after Phase 1).
- **Re-verify on every vLLM upgrade.** The extension point has churn (V0 API removed in v0.17.0; Model Runner V2 gap; spec-decode incompatibility). Pin exact versions in code and record them in every experiment.

## 2. Licensing rules

- **Never copy code from `eth-sri/unified-watermarking`** — it has no license (all rights reserved by default). Design/architecture reference only.
- Port algorithm logic only from Apache-2.0 sources: `huggingface/transformers`, `google-deepmind/synthid-text`, `THU-BPM/MarkLLM`. Attribute in file headers.
- `dapurv5/vLLM-Watermark` has inconsistent license metadata — don't copy from it either (and its approach is rejected anyway; see [`docs/technical.md`](docs/technical.md)).

## 3. Secrets and safety

- The repo-root `aws` file and `cluster/` directory contain **live credentials** — both are gitignored. Never commit, print, echo, or paste their contents. Never weaken `.gitignore`.
- Watermark keys: env vars / mounted Secrets only. Never hardcode, never log, never commit — including in `EXPERIMENTS.md` output captures (redact before committing).
- The GPU node is billable: `./scripts/scale-gpu.sh 1` to start, **`./scripts/scale-gpu.sh 0` when done** — check this before ending any work session.
- `KUBECONFIG=cluster/auth/kubeconfig` for cluster access.

## 4. Content policy for this repo

- No personal names, no internal communications, no customer identifiers. Refer to roles ("the enterprise", "counsel", "product management") only.
- Regulatory statements: quote-and-cite only, per [`docs/quotes.md`](docs/quotes.md). This repo is not legal advice and must not silently paraphrase legal text — paraphrases drift.

## 5. Environment facts

- **Local workstation cannot run vLLM** (Python 3.14.4, no GPU; vLLM v0.26.0 pins torch==2.11.0/transformers>=5.5.3). Local is fine for detector math, unit tests, docs. All vLLM execution happens on the cluster ([`docs/cluster.md`](docs/cluster.md)): OpenShift 4.20, `ocp-ai`, g5.xlarge GPU MachineSet (A10G 24GB, fits ≤8B models comfortably).
- RHOAI target: 3.4.x (ships vLLM 0.17.1–0.18.0 — V1 plugin API only; V0-era code paths are dead).

## 6. Workflow

- Work [`docs/implementation.md`](docs/implementation.md) phases **in order**; each has acceptance criteria — meet them before moving on, or record precisely why not in `EXPERIMENTS.md`.
- Code layout: `src/vllm_watermark/` (package), `detector/` (service), `deploy/` (Dockerfile + manifests), `benchmarks/` (scripts), `research/` (read-only history — don't edit).
- Commits: small, imperative subject, body says what was verified and how. Don't commit generated model outputs except as summarized tables in `EXPERIMENTS.md`.
- When blocked on something engineering can't resolve (support policy, legal scope), record it under "Out of scope" status in `EXPERIMENTS.md` and continue with the next unblocked step — don't improvise answers to non-engineering questions.
