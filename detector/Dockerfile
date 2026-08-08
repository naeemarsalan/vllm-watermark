# Watermark detector service image (Task C3).
#
# Built by an OpenShift BuildConfig using the Docker strategy with a Binary
# source (spec.source.type: Binary, spec.strategy.type: Docker,
# spec.strategy.dockerStrategy.dockerfilePath: detector/Dockerfile) -- see
# deploy/phase3/detector-build.yaml + deploy/phase3/README.md for the exact
# `oc start-build` invocation and field citations (OpenShift 4.20 docs,
# openshift/openshift-docs repo, branch enterprise-4.20, fetched 2026-08-08:
# modules/builds-binary-source.adoc, modules/builds-strategy-dockerfile-path.adoc,
# rest_api/workloads_apis/buildconfig-build-openshift-io-v1.adoc).
#
# BUILD CONTEXT IS THE REPO ROOT (or a curated staging copy of it -- see the
# README's "Build context safety" section for why plain `--from-dir=.` from a
# checkout containing the live, gitignored `cluster/` credentials directory is
# NOT what the runbook actually recommends), not `detector/` alone: the
# vllm-watermark wheel this image installs lives at `dist/*.whl`, outside this
# directory, and Buildah's COPY cannot reach outside the build context. All
# COPY paths below are therefore written relative to the REPO ROOT.
#
# Base image: python:3.12-slim (task-specified). Pinned by tag, not digest --
# unlike the vLLM base image in deploy/phase0 (large, rebuilt rarely), this
# image is expected to be rebuilt on every vllm_watermark wheel change, and
# Docker Hub retags `3.12-slim` frequently with Debian security patches. If a
# fully reproducible pin is later required, resolve the digest actually
# pulled via `oc describe build detector-<N> | grep -i "docker image"` after
# a build and switch this line to `python:3.12-slim@sha256:<digest>`.
FROM python:3.12-slim

# --- torch (CPU wheel) ----------------------------------------------------
# EXECUTED verification (2026-08-08, see EXPERIMENTS.md this task's entry):
#   1. `curl -s https://download.pytorch.org/whl/cpu/torch/` lists
#      torch-2.9.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl -- the exact
#      (version, python tag, platform) triple this base image needs.
#   2. `curl -s https://pypi.org/pypi/torch/2.9.1/json` shows torch's
#      unconditional (marker-free) dependency on `jinja2`, and
#      `curl -s https://download.pytorch.org/whl/cpu/MarkupSafe/` returns ZERO
#      package links -- the CPU wheel index does not host `MarkupSafe`
#      (jinja2's own dependency). A plain
#      `pip install --index-url https://download.pytorch.org/whl/cpu torch`
#      (no fallback index) would therefore fail to resolve at that transitive
#      dependency. `--extra-index-url https://pypi.org/simple` supplies the
#      pure-Python packages (jinja2, MarkupSafe, sympy, networkx, fsspec,
#      filelock, typing-extensions, mpmath, setuptools) the CPU wheel index
#      doesn't carry, while `--index-url` (searched first) still ensures the
#      `torch` package itself resolves to the CPU build, never PyPI's default
#      CUDA-bundled `torch` wheel (confirmed via the same PyPI JSON metadata:
#      the default `torch` requires 17 unconditional `nvidia-cu12-*`
#      packages on linux/x86_64 -- exactly what pinning to the CPU index
#      avoids).
# Version 2.9.1 matches the exact torch build every KGW/SynthID number in
# this repo's EXPERIMENTS.md was computed against (`pip show torch` on the
# local workstation).
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.9.1

# --- detector service HTTP/crypto/tokenizer stack -------------------------
COPY detector/requirements.txt /tmp/detector-requirements.txt
RUN pip install --no-cache-dir -r /tmp/detector-requirements.txt \
    && rm /tmp/detector-requirements.txt

# --- the vllm_watermark wheel (built by deploy/phase0/build-wheel.sh) -----
# --no-deps: the wheel's only runtime dependency is `torch>=2.1`
# (pyproject.toml), already satisfied by the pinned CPU build above --
# installing it again unpinned could silently swap in PyPI's default CUDA
# `torch`. The glob is version-number-agnostic (matches whatever
# build-wheel.sh produced, e.g. vllm_watermark-0.1.0.dev0-py3-none-any.whl).
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && rm -f /tmp/*.whl

# --- application code -------------------------------------------------
# Only detector/app.py is copied -- NOT detector/tests/ (local-dev-only
# pytest fixtures, not runtime code; see detector/tests/conftest.py's own
# docstring) and not this Dockerfile itself.
WORKDIR /app
COPY detector/app.py /app/detector/app.py

# --- non-root-friendly runtime configuration -------------------------
# Deliberately NO `USER` directive: this image runs under OpenShift's
# restricted-v2 SCC, which assigns an arbitrary UID (primary group 0) to the
# container at admission time regardless of any USER set in the image -- the
# same convention already established for the vLLM pods in
# deploy/phase0/vllm-baseline-pod.yaml (see that file's header comment).
# HF_HOME/HOME/XDG_*_HOME all point at /tmp/hf: the base image's default
# $HOME (/root) is not writable under an arbitrary UID, and
# `AutoTokenizer.from_pretrained` (detector/app.py's `_lifespan`) needs a
# writable cache directory to download the tokenizer on first start.
# deploy/phase3/detector-deploy.yaml backs /tmp/hf with a dedicated emptyDir
# at runtime, per this task's brief; the `chmod 0777` below is defense in
# depth for anyone running this image directly (`podman run`, no emptyDir)
# rather than something the emptyDir mount depends on -- Kubernetes emptyDir
# directories are themselves created world-writable, and plain container
# /tmp is already mode 1777 on this base image regardless.
ENV HF_HOME=/tmp/hf \
    HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/hf/.cache \
    XDG_CONFIG_HOME=/tmp/hf/.config \
    PYTHONUNBUFFERED=1
RUN mkdir -p /tmp/hf && chmod -R 0777 /tmp/hf

EXPOSE 8000

# `--app-dir /app`, not a bare reliance on `WORKDIR`: uvicorn is invoked here
# as an installed console-script entry point, which -- unlike `python -m
# uvicorn` -- does NOT add the current working directory to `sys.path`
# automatically. `--app-dir` is uvicorn's own documented mechanism for this.
# It lets the import string `detector.app:app` resolve `detector` as a PEP
# 420 implicit namespace package rooted at /app (detector/app.py has no
# `__init__.py` -- verified by reading detector/tests/conftest.py's own
# docstring, which documents the same fact and the same two invocation forms
# this Dockerfile picks the second of: "`uvicorn app:app` (from within the
# `detector/` directory) or `uvicorn detector.app:app` (from the repo
# root)"). Single worker (no `--workers`): the tokenizer and torch state
# `_lifespan` loads at startup would otherwise be duplicated per worker
# process, and this task's resource sizing (deploy/phase3/detector-deploy.yaml,
# cpu 1 / mem 2Gi request, 4Gi limit) is sized for exactly one.
CMD ["uvicorn", "detector.app:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8000"]
