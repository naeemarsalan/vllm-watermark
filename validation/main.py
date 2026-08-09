"""Deployable D10 validation gateway entry point.

``python -m validation.main`` constructs the production HTTP adapters only
after every required endpoint, token, and managed-NeMo configuration field is
present and valid.  It intentionally does not log configuration values.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn

from validation.gateway import ConfigurationError, GatewayConfig, GatewayService, ManagedNeMoAdapter, PendingValidationBroker, create_app
from validation.http_clients import DirectDetectorClient, HttpClientSettings, ManagedNeMo021Client, OpenAIUpstreamClient


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} is required")
    return value.strip()


def _optional_token(env: Mapping[str, str], name: str) -> str | None:
    """Return an internal bearer token only when explicitly configured."""
    if name not in env:
        return None
    value = env.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be non-empty when configured")
    return value.strip()


def _positive_int(value: str, name: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ConfigurationError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ConfigurationError(f"{name} must be a finite positive number")
    return parsed


def _boolean(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be an explicit boolean")


@dataclass(frozen=True)
class RuntimeConfig:
    upstream_url: str
    upstream_token: str | None = field(default=None, repr=False)
    detector_url: str = ""
    detector_token: str | None = field(default=None, repr=False)
    nemo_url: str = ""
    nemo_token: str = field(default="", repr=False)
    nemo_config_id: str = ""
    nemo_model: str = ""
    http_settings: HttpClientSettings = field(default_factory=HttpClientSettings)
    host: str = "0.0.0.0"
    port: int = 8080

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "RuntimeConfig":
        env = os.environ if env is None else env
        settings = HttpClientSettings(
            timeout_seconds=_positive_float(env.get("VALIDATION_HTTP_TIMEOUT_SECONDS", "10"), "VALIDATION_HTTP_TIMEOUT_SECONDS"),
            max_response_bytes=_positive_int(env.get("VALIDATION_MAX_RESPONSE_BYTES", "1048576"), "VALIDATION_MAX_RESPONSE_BYTES"),
            # System trust is the default; an optional mounted bundle adds a
            # private CA. Verification may never be disabled.
            verify_tls=_boolean(env.get("VALIDATION_TLS_VERIFY", "true"), "VALIDATION_TLS_VERIFY"),
            tls_ca_bundle=env.get("VALIDATION_TLS_CA_BUNDLE"),
        )
        return cls(
            upstream_url=_required(env, "VALIDATION_UPSTREAM_URL"),
            upstream_token=_optional_token(env, "VALIDATION_UPSTREAM_TOKEN"),
            detector_url=_required(env, "VALIDATION_DETECTOR_URL"),
            detector_token=_optional_token(env, "VALIDATION_DETECTOR_TOKEN"),
            nemo_url=_required(env, "VALIDATION_NEMO_URL"),
            nemo_token=_required(env, "VALIDATION_NEMO_TOKEN"),
            nemo_config_id=_required(env, "VALIDATION_NEMO_CONFIG_ID"),
            nemo_model=_required(env, "VALIDATION_NEMO_MODEL"),
            http_settings=settings,
            host=env.get("VALIDATION_HOST", "0.0.0.0"),
            port=_positive_int(env.get("VALIDATION_PORT", "8080"), "VALIDATION_PORT"),
        )


def create_runtime_app(
    env: Mapping[str, str] | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Any:
    """Build the FastAPI gateway with concrete, bounded HTTP transports.

    ``transport`` exists solely for isolated tests; the command-line entry
    point leaves it as ``None`` and uses normal network transport.
    """
    gateway_config = GatewayConfig.from_environment(env)
    runtime = RuntimeConfig.from_environment(env)
    upstream = OpenAIUpstreamClient(runtime.upstream_url, runtime.upstream_token, runtime.http_settings, transport)
    detector = DirectDetectorClient(runtime.detector_url, runtime.detector_token, runtime.http_settings, transport)
    broker = PendingValidationBroker(detector)
    nemo = ManagedNeMo021Client(
        runtime.nemo_url,
        runtime.nemo_token,
        runtime.nemo_config_id,
        runtime.nemo_model,
        runtime.http_settings,
        transport,
    )
    return create_app(GatewayService(gateway_config, upstream, ManagedNeMoAdapter(nemo, broker), broker=broker))


def main() -> None:
    """Run one process so the pending broker and SQLite sampler stay coherent."""
    runtime = RuntimeConfig.from_environment()
    app = create_runtime_app()
    uvicorn.run(app, host=runtime.host, port=runtime.port, log_level="info", workers=1)


if __name__ == "__main__":  # pragma: no cover - exercised by deployment.
    main()
