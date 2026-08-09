"""Continuous-validation gateway primitives.

This package deliberately has no dependency on the watermark key material.  It
only carries non-secret watermark metadata and a SHA-256 digest across the
validation boundary.
"""

from .gateway import (
    GatewayConfig,
    GatewayService,
    ManagedValidationAdapter,
    ValidationOutcome,
    create_app,
)

__all__ = [
    "GatewayConfig",
    "GatewayService",
    "ManagedValidationAdapter",
    "ValidationOutcome",
    "create_app",
]
