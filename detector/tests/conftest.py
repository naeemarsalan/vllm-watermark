"""Shared pytest setup for `detector/tests/test_service.py`.

Bootstraps `sys.path` so tests can run directly with
`/usr/bin/python3 -m pytest detector/tests/` from the repo root WITHOUT an
editable/wheel install of either `vllm_watermark` (the pure-python package
under `src/`) or `detector.app` itself -- mirrors the exact pattern already
used by the top-level `tests/conftest.py` for the same reason (AGENTS.md
"pip install --user of pure-python test deps ... IS allowed; never install
vllm" -- installing OUR OWN package is unnecessary when a path insert does
the same job with zero side effects on the environment).

In the real container deployment (see `detector/README.md`), the built
`vllm_watermark` wheel is `pip install`ed properly and `detector/app.py`
runs as `uvicorn app:app` from within the `detector/` directory (or
`uvicorn detector.app:app` from the repo root) -- this sys.path bootstrap
is a LOCAL TEST convenience only, not something app.py itself relies on.

All key material anywhere in this test suite is an obviously-dummy test
value (AGENTS.md #3 secrets policy) -- `"aa" * 16` is not used
anywhere outside `detector/tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "src"
_DETECTOR = _REPO_ROOT / "detector"

for _p in (_SRC, _DETECTOR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
