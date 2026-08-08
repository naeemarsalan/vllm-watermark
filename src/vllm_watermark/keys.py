"""Watermark key loading.

Keys never come from source code, never have a hardcoded fallback, and are
never logged/printed in cleartext. See AGENTS.md #3 (secrets) and CLAUDE.md
task instructions: "watermark keys come from env vars only. Never hardcode
a production key. ... Never print key material."

Env var formats supported (checked in this order):

1. WATERMARK_KEYS = "keyid1:hexsecret1,keyid2:hexsecret2,..."
   -- a comma-separated list of "key_id:hex_secret" pairs. Use this form
   to register multiple keys (e.g. for rotation) under distinct key_ids.

2. WATERMARK_KEY = "hexsecret"  (+ optional WATERMARK_KEY_ID, default
   "default") -- a convenience single-key form.

If neither is set, loading raises RuntimeError -- there is intentionally
no silent default key.

`hex_secret` is an arbitrary-length hex string (an even number of hex
digits, decoded with bytes.fromhex). The KGW hash_key actually used is
derived as:

    hash_key = int.from_bytes(sha256(secret_bytes)[:8], "big")

i.e. the first 8 bytes (64 bits) of SHA-256(secret_bytes), read as a big
endian unsigned integer. This gives a full-range uniformly-derived 64-bit
hash_key from an arbitrary-length secret, without ever using the raw
secret bytes as the hash_key directly.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

_DEFAULT_KEY_ID = "default"


@dataclass(frozen=True)
class WatermarkKey:
    """A loaded watermark key. repr/str are redacted -- never expose secret
    material via logging, exceptions, or debugger output."""

    key_id: str
    hash_key: int

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"WatermarkKey(key_id={self.key_id!r}, hash_key=<redacted>)"

    __str__ = __repr__


def _derive_hash_key(secret_bytes: bytes) -> int:
    digest = hashlib.sha256(secret_bytes).digest()
    return int.from_bytes(digest[:8], "big")


def _parse_hex_secret(key_id: str, hex_secret: str) -> WatermarkKey:
    hex_secret = hex_secret.strip()
    try:
        secret_bytes = bytes.fromhex(hex_secret)
    except ValueError as exc:
        raise ValueError(
            f"secret for key_id={key_id!r} is not valid hex "
            "(WATERMARK_KEYS/WATERMARK_KEY must be hex-encoded secrets)"
        ) from exc
    if not secret_bytes:
        raise ValueError(f"secret for key_id={key_id!r} is empty")
    return WatermarkKey(key_id=key_id, hash_key=_derive_hash_key(secret_bytes))


def load_keys(env: "os._Environ[str] | dict[str, str] | None" = None) -> dict[str, WatermarkKey]:
    """Load all configured watermark keys from the environment.

    Raises RuntimeError if no key material is configured at all -- callers
    must not fall back to a default key.
    """
    if env is None:
        env = os.environ

    keys_blob = env.get("WATERMARK_KEYS")
    if keys_blob:
        keys: dict[str, WatermarkKey] = {}
        for entry in keys_blob.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(
                    f"malformed WATERMARK_KEYS entry {entry!r}; expected 'key_id:hexsecret'"
                )
            key_id, hex_secret = entry.split(":", 1)
            key_id = key_id.strip()
            if not key_id:
                raise ValueError(f"malformed WATERMARK_KEYS entry {entry!r}; empty key_id")
            if key_id in keys:
                raise ValueError(f"duplicate key_id {key_id!r} in WATERMARK_KEYS")
            keys[key_id] = _parse_hex_secret(key_id, hex_secret)
        if not keys:
            raise RuntimeError("WATERMARK_KEYS is set but contains no key entries")
        return keys

    single_secret = env.get("WATERMARK_KEY")
    if single_secret:
        key_id = env.get("WATERMARK_KEY_ID", _DEFAULT_KEY_ID).strip() or _DEFAULT_KEY_ID
        return {key_id: _parse_hex_secret(key_id, single_secret)}

    raise RuntimeError(
        "No watermark key configured: set WATERMARK_KEYS ('keyid:hexsecret,...') "
        "or WATERMARK_KEY (+ optional WATERMARK_KEY_ID). Refusing to fall back "
        "to a default key."
    )


def load_key(
    key_id: "str | None" = None,
    env: "os._Environ[str] | dict[str, str] | None" = None,
) -> WatermarkKey:
    """Load a single key by id (default: WATERMARK_KEY_ID env var, else
    "default"). Raises KeyError if that key_id was not configured."""
    if env is None:
        env = os.environ
    keys = load_keys(env)
    if key_id is None:
        key_id = env.get("WATERMARK_KEY_ID", _DEFAULT_KEY_ID).strip() or _DEFAULT_KEY_ID
    if key_id not in keys:
        raise KeyError(
            f"key_id={key_id!r} not found among configured keys: {sorted(keys)}"
        )
    return keys[key_id]
