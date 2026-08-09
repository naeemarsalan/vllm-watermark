"""Bounded HTTP transports for the D10 validation gateway.

These clients deliberately keep generated text and bearer tokens out of error
messages and object representations.  They implement only the contracts used by
``validation.gateway``; deployment-specific configuration lives in
``validation.main``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import ssl
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from validation.gateway import (
    ConfigurationError,
    RetryableValidationError,
    TerminalValidationError,
    ValidationRequest,
    Verdict,
    _response_id,
)


_MAX_RESPONSE_BYTES_DEFAULT = 1_048_576
_DETECTOR_PATH = "/v1/watermark/detect"
_NEMO_CHECKS_PATH = "/v1/guardrail/checks"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_NEMO_SCHEMES = frozenset({"kgw", "synthid"})
_NEMO_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _required(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} is required")
    return value.strip()


def _optional_token(value: str | None, name: str) -> str | None:
    """Validate a configured token, while allowing no auth for internals."""
    if value is None:
        return None
    return _required(value, name)


def _base_url(value: str, name: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigurationError(f"{name} must not contain credentials, a query, or a fragment")
    return value.rstrip("/")


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url}{path}"


def _canonical_uuid(value: str) -> str:
    if not isinstance(value, str):
        raise TerminalValidationError("validation id is not a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise TerminalValidationError("validation id is not a canonical UUID") from exc
    if str(parsed) != value:
        raise TerminalValidationError("validation id is not a canonical UUID")
    return value


def _finite_positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be a finite positive number")
    return value


@dataclass(frozen=True)
class HttpClientSettings:
    """Shared transport limits with mandatory TLS certificate verification."""

    timeout_seconds: float = 10.0
    max_response_bytes: int = _MAX_RESPONSE_BYTES_DEFAULT
    verify_tls: bool = True
    tls_ca_bundle: str | None = None

    def __post_init__(self) -> None:
        _finite_positive(self.timeout_seconds, "VALIDATION_HTTP_TIMEOUT_SECONDS")
        if not isinstance(self.max_response_bytes, int) or isinstance(self.max_response_bytes, bool) or self.max_response_bytes < 1:
            raise ConfigurationError("VALIDATION_MAX_RESPONSE_BYTES must be a positive integer")
        if self.verify_tls is not True:
            raise ConfigurationError("VALIDATION_TLS_VERIFY=false is not permitted")
        if self.tls_ca_bundle is not None:
            if not isinstance(self.tls_ca_bundle, str) or not self.tls_ca_bundle.strip():
                raise ConfigurationError("VALIDATION_TLS_CA_BUNDLE must be a non-empty path")
            try:
                ssl.create_default_context(cafile=self.tls_ca_bundle)
            except (OSError, ssl.SSLError) as exc:
                raise ConfigurationError("VALIDATION_TLS_CA_BUNDLE is not a usable CA bundle") from exc


class _BoundedHttpClient:
    """Small, non-logging HTTP base with a hard response-size limit."""

    def __init__(
        self,
        *,
        settings: HttpClientSettings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Concrete clients are frozen dataclasses so their secret-bearing
        # configuration cannot be accidentally mutated after startup.
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_transport", transport)

    async def _post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            verify = ssl.create_default_context(cafile=self._settings.tls_ca_bundle)
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.timeout_seconds),
                verify=verify,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                async with client.stream("POST", url, headers=dict(headers), json=dict(payload)) as response:
                    if response.status_code in {408, 425, 429} or response.status_code >= 500:
                        raise RetryableValidationError("remote service returned a retryable HTTP status")
                    if response.status_code < 200 or response.status_code >= 300:
                        raise TerminalValidationError("remote service returned a non-success HTTP status")
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(raw) + len(chunk) > self._settings.max_response_bytes:
                            raise TerminalValidationError("remote service response exceeds configured size limit")
                        raw.extend(chunk)
        except (RetryableValidationError, TerminalValidationError):
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableValidationError("remote service transport failed") from exc

        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise TerminalValidationError("remote service returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise TerminalValidationError("remote service returned a non-object JSON response")
        return parsed


@dataclass(frozen=True)
class OpenAIUpstreamClient(_BoundedHttpClient):
    """OpenAI-compatible upstream proxy transport."""

    base_url: str
    bearer_token: str | None = field(default=None, repr=False)
    settings: HttpClientSettings = field(default_factory=HttpClientSettings)
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _base_url(self.base_url, "VALIDATION_UPSTREAM_URL"))
        object.__setattr__(self, "bearer_token", _optional_token(self.bearer_token, "VALIDATION_UPSTREAM_TOKEN"))
        _BoundedHttpClient.__init__(self, settings=self.settings, transport=self.transport)

    async def complete(self, endpoint: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if endpoint not in {"/v1/completions", "/v1/chat/completions"}:
            raise TerminalValidationError("unsupported OpenAI upstream endpoint")
        headers = {"Accept": "application/json"}
        if self.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return await self._post_json(
            _endpoint(self.base_url, endpoint),
            headers=headers,
            payload=request,
        )


@dataclass(frozen=True)
class DirectDetectorClient(_BoundedHttpClient):
    """Direct detector client with strict response correlation validation."""

    url: str
    bearer_token: str | None = field(default=None, repr=False)
    settings: HttpClientSettings = field(default_factory=HttpClientSettings)
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        url = _base_url(self.url, "VALIDATION_DETECTOR_URL")
        if not url.endswith(_DETECTOR_PATH):
            raise ConfigurationError(f"VALIDATION_DETECTOR_URL must end with {_DETECTOR_PATH}")
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "bearer_token", _optional_token(self.bearer_token, "VALIDATION_DETECTOR_TOKEN"))
        _BoundedHttpClient.__init__(self, settings=self.settings, transport=self.transport)

    async def detect(self, request: ValidationRequest) -> Verdict:
        validation_id = _canonical_uuid(request.validation_id)
        response_id = _response_id(request.response_id)
        expected_digest = hashlib.sha256(request.content.encode("utf-8")).hexdigest()
        if expected_digest != request.content_sha256:
            raise TerminalValidationError("validation request content digest is inconsistent")
        headers = {
            "Accept": "application/json",
            "X-Watermark-Validation-Id": validation_id,
        }
        if self.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        payload = await self._post_json(
            self.url,
            headers=headers,
            payload={
                "text": request.content,
                "validation_id": validation_id,
                "response_id": response_id,
                "scheme": request.scheme,
                "key_id": request.key_id,
            },
        )
        expected = {
            "validation_id": validation_id,
            "response_id": response_id,
            "content_sha256": request.content_sha256,
            "scheme": request.scheme,
            "key_id": request.key_id,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise TerminalValidationError("detector response correlation metadata does not match")
        verdict = payload.get("verdict")
        if not isinstance(verdict, bool):
            raise TerminalValidationError("detector response verdict is not a boolean")
        return "watermarked" if verdict else "clean"


@dataclass(frozen=True)
class ManagedNeMo021Client(_BoundedHttpClient):
    """RHOAI-managed NeMo 0.21 guardrail-checks client.

    The endpoint accepts an assistant message to invoke output rails.  The five
    non-content correlation fields are sent both in ``guardrails.context`` and
    as bounded ``X-Watermark-*`` headers for the managed action; no detector
    verdict is inferred from the NeMo response.
    """

    base_url: str
    bearer_token: str = field(repr=False)
    config_id: str = ""
    model: str = ""
    settings: HttpClientSettings = field(default_factory=HttpClientSettings)
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        base_url = _base_url(self.base_url, "VALIDATION_NEMO_URL")
        if base_url.endswith(_NEMO_CHECKS_PATH):
            raise ConfigurationError("VALIDATION_NEMO_URL must be a base URL, not the guardrail checks endpoint")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "bearer_token", _required(self.bearer_token, "VALIDATION_NEMO_TOKEN"))
        object.__setattr__(self, "config_id", _required(self.config_id, "VALIDATION_NEMO_CONFIG_ID"))
        object.__setattr__(self, "model", _required(self.model, "VALIDATION_NEMO_MODEL"))
        _BoundedHttpClient.__init__(self, settings=self.settings, transport=self.transport)

    async def validate(self, text: str, context: Mapping[str, str]) -> str:
        required_context = {
            "watermark_validation_id",
            "watermark_response_id",
            "watermark_content_sha256",
            "watermark_scheme",
            "watermark_key_id",
        }
        if set(context) != required_context:
            raise TerminalValidationError("managed NeMo correlation context is malformed")
        validation_id = _canonical_uuid(context["watermark_validation_id"])
        response_id = _response_id(context["watermark_response_id"])
        digest = context["watermark_content_sha256"]
        scheme = context["watermark_scheme"]
        key_id = context["watermark_key_id"]
        if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
            raise TerminalValidationError("managed NeMo correlation digest is malformed")
        if digest != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise TerminalValidationError("managed NeMo correlation digest is inconsistent")
        if not isinstance(scheme, str) or scheme not in _NEMO_SCHEMES:
            raise TerminalValidationError("managed NeMo correlation scheme is malformed")
        if not isinstance(key_id, str) or not _NEMO_KEY_ID.fullmatch(key_id):
            raise TerminalValidationError("managed NeMo correlation key id is malformed")
        payload = await self._post_json(
            _endpoint(self.base_url, _NEMO_CHECKS_PATH),
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "Accept": "application/json",
                "X-Watermark-Validation-Id": validation_id,
                "X-Watermark-Response-Id": response_id,
                "X-Watermark-Content-Sha256": digest,
                "X-Watermark-Scheme": scheme,
                "X-Watermark-Key-Id": key_id,
            },
            payload={
                "model": self.model,
                "messages": [{"role": "assistant", "content": text}],
                "guardrails": {"config_id": self.config_id, "context": dict(context)},
            },
        )
        status = payload.get("status")
        if status not in {"blocked", "success"}:
            raise TerminalValidationError("managed NeMo returned an invalid action state")
        # This is the executed RHOAI-managed 0.21 response nesting.  Do not
        # accept legacy ``message_results`` or alternate rail locations: an
        # outer status without both matching named rail summaries cannot prove
        # which rail produced the decision.
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) != 1:
            raise TerminalValidationError("managed NeMo messages is malformed")
        message = messages[0]
        if not isinstance(message, Mapping):
            raise TerminalValidationError("managed NeMo message is malformed")
        if type(message.get("index")) is not int or message["index"] != 0 or message.get("role") != "assistant":
            raise TerminalValidationError("managed NeMo message identity is malformed")
        rails = message.get("rails")
        if not isinstance(rails, Mapping):
            raise TerminalValidationError("managed NeMo rails result is malformed")
        message_rail = rails.get("watermark check")
        if not isinstance(message_rail, Mapping) or set(message_rail) != {"status"} or message_rail.get("status") != status:
            raise TerminalValidationError("managed NeMo message watermark check does not match top-level status")
        rails_status = payload.get("rails_status")
        if not isinstance(rails_status, Mapping):
            raise TerminalValidationError("managed NeMo aggregate rails status is malformed")
        aggregate_rail = rails_status.get("watermark check")
        if not isinstance(aggregate_rail, Mapping) or set(aggregate_rail) != {"status"} or aggregate_rail.get("status") != status:
            raise TerminalValidationError("managed NeMo aggregate watermark check does not match top-level status")
        return status
