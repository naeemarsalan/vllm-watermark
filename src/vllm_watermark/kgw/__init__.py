"""KGW (Kirchenbauer-Geiping-Wen) green-list watermark: core + detector.

This subpackage does not import vllm or transformers at module level (see
core.py / detector.py docstrings for the exact upstream transformers
sources each was ported from, with attribution + license headers).
"""

from vllm_watermark.kgw.core import KGWConfig, greenlist_ids
from vllm_watermark.kgw.detector import DetectionResult, detect_text, score_token_ids

__all__ = [
    "KGWConfig",
    "greenlist_ids",
    "DetectionResult",
    "score_token_ids",
    "detect_text",
]
