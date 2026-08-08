"""vllm_watermark: decode-time text watermarking (KGW today; SynthID planned).

This top-level package does NOT import vllm or transformers -- vllm is not
installed on the local dev workstation and must not be a hard import-time
dependency of the pure-algorithm code (see AGENTS.md / CLAUDE.md task
constraints); transformers is optional (only needed for
kgw.detector.detect_text's tokenizer argument, and lazily inside cli.py).
"""

__version__ = "0.1.0.dev0"
