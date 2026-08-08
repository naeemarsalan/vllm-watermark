#!/usr/bin/env bash
# Build the vllm_watermark wheel used by the wheel-injection sequence in
# vllm-watermark-pod.yaml / README.md.
#
# TESTED 2026-08-08 on the local workstation (Python 3.14.4, pip 25.1.1,
# no `build` or `wheel` packages installed system-wide -- `pip wheel`
# does not require either; it uses PEP 517 build isolation against the
# `[build-system]` table in pyproject.toml, which is `setuptools>=68` /
# `setuptools.build_meta` in this repo):
#
#   $ /usr/bin/python3 -m pip wheel --no-deps -w dist .
#   ...
#   Successfully built vllm-watermark
#   $ ls dist/
#   vllm_watermark-0.1.0.dev0-py3-none-any.whl
#
# Pure-Python wheel (no compiled extensions -- only runtime dep is torch,
# already present in the vllm-openai image), so this ONE build, from
# ANY Python 3.11+/3.14 interpreter, is what gets `oc cp`'d into the
# watermark pod regardless of that pod's own Python minor version -- see
# vllm-watermark-pod.yaml's header comment for why that's safe here.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"

python_bin="${PYTHON:-/usr/bin/python3}"

if [[ ! -f pyproject.toml ]]; then
  echo "pyproject.toml not found at repo root ($repo_dir) -- nothing to build yet." >&2
  exit 1
fi

mkdir -p dist

if "$python_bin" -m pip wheel --no-deps -w dist .; then
  echo "Wheel built: $(ls dist/vllm_watermark-*.whl)"
elif "$python_bin" -m build --version >/dev/null 2>&1; then
  # Fallback path, not exercised in the 2026-08-08 test run above (pip
  # wheel succeeded there) -- kept for environments where `pip wheel`'s
  # build-isolation step can't reach PyPI but a pre-installed `build` can.
  "$python_bin" -m build --wheel --outdir dist .
  echo "Wheel built (via python -m build): $(ls dist/vllm_watermark-*.whl)"
else
  echo "Both 'pip wheel' and 'python -m build' failed -- see output above." >&2
  exit 1
fi
