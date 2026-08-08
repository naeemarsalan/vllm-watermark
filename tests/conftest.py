"""Make the src-layout package importable when running pytest from the repo root
without an editable install (the local workstation cannot pip-install vllm, and
an editable install is not required just to test the vLLM-free modules)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
