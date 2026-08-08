# SPDX-License-Identifier: Apache-2.0
"""SynthID-Text tournament-sampling watermark: core + detector.

This subpackage does not import vllm or transformers at module level (see
core.py / detector.py docstrings for the exact upstream transformers /
google-deepmind/synthid-text sources each was ported from, with
attribution + license headers). Mirrors vllm_watermark.kgw's package
shape/import discipline.
"""

from vllm_watermark.synthid.core import (
    DEFAULT_SYNTHID_DEPTH,
    SYNTHID_KEY_LABEL,
    SynthIDConfig,
    expected_mean_g_value,
    g_values,
    process_scores_row,
)
from vllm_watermark.synthid.detector import (
    DetectionResult,
    detect_text,
    score_token_ids_mean,
    score_token_ids_weighted_mean,
)

__all__ = [
    "DEFAULT_SYNTHID_DEPTH",
    "SYNTHID_KEY_LABEL",
    "SynthIDConfig",
    "expected_mean_g_value",
    "g_values",
    "process_scores_row",
    "DetectionResult",
    "score_token_ids_mean",
    "score_token_ids_weighted_mean",
    "detect_text",
]
