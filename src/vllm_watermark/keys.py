"""Watermark key loading.

Keys never come from source code, never have a hardcoded fallback, and are
never logged/printed in cleartext. See AGENTS.md #3: watermark keys come
from environment variables or mounted Secrets, are never hardcoded, and
key material is never printed.

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

SynthID subkey derivation (`WatermarkKey.derive_subkeys`, added in Phase 2)
------------------------------------------------------------------------------
SynthID-Text needs a *list* of integer keys, one per tournament layer/depth
(default depth 30 -- see `vllm_watermark.synthid.core.DEFAULT_SYNTHID_DEPTH`),
not a single hash_key. Re-deriving those subkeys from `secret_bytes` directly
would require retaining the raw secret for the life of the process, which
this module has always refused to do (only a derived `hash_key` was kept).
Instead, `WatermarkKey` additionally retains `secret_digest: bytes` --
`sha256(secret_bytes).digest()` (32 raw bytes, NOT the secret itself) -- and
`derive_subkeys(n, label)` re-derives each subkey from that digest:

    keys[i] = int.from_bytes(sha256(secret_digest + label + i.to_bytes(4, "big")).digest()[:4], "big")

for i in range(n), giving n independent 32-bit unsigned ints (small enough
to match the magnitude of the `keys=[654, 400, 836, ...]` example in
transformers' own SynthIDTextWatermarkingConfig docstring, and cheap to fold
into `vllm_watermark.synthid.core`'s int64 LCG hashing). `label` namespaces
different subkey purposes drawn from the same secret (e.g. b"synthid-keys")
so two different derived-key lists never collide even if `n` happens to
match by coincidence.

IMPORTANT -- secret_digest is secret-equivalent material, NOT a public
fingerprint: SHA-256 is deterministic and un-salted here, so anyone who
learns `secret_digest` can reproduce every `derive_subkeys` output (and, if
they can also brute-force a low-entropy secret against the digest, recover
`hash_key` and the secret itself). It MUST be treated with exactly the same
handling rules as `hash_key` -- never logged, never printed, never included
in `EXPERIMENTS.md` captures. `WatermarkKey.__repr__`/`__str__` redact it
for the same reason they redact `hash_key`.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

_DEFAULT_KEY_ID = "default"


@dataclass(frozen=True)
class WatermarkKey:
    """A loaded watermark key. repr/str are redacted -- never expose secret
    material via logging, exceptions, or debugger output.

    `secret_digest` (32 raw bytes, sha256(secret_bytes)) is retained
    alongside `hash_key` solely so `derive_subkeys()` can deterministically
    re-derive SynthID's per-layer key list without this module ever holding
    onto the raw secret bytes themselves. It is exactly as sensitive as
    `hash_key` -- see module docstring "SynthID subkey derivation" -- and is
    redacted identically.
    """

    key_id: str
    hash_key: int
    secret_digest: bytes

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"WatermarkKey(key_id={self.key_id!r}, hash_key=<redacted>, secret_digest=<redacted>)"

    __str__ = __repr__

    def derive_subkeys(self, n: int, label: bytes) -> tuple[int, ...]:
        """Deterministically derive `n` independent 32-bit subkeys from this
        key's secret_digest, namespaced by `label`.

        keys[i] = int.from_bytes(sha256(secret_digest + label + i.to_bytes(4, "big")).digest()[:4], "big")

        Used by `vllm_watermark.synthid.core` to build the per-tournament-
        layer key list SynthID needs from the same secret material that
        drives this key's KGW `hash_key` -- see module docstring "SynthID
        subkey derivation" for why this re-derives from a stored digest
        rather than requiring the raw secret to be kept around.

        Deterministic: the same (secret, n, label) always yields the same
        tuple, so generation-time and detection-time callers (given the
        same configured key and the same label) always agree.
        """
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
        if not isinstance(label, (bytes, bytearray)):
            raise TypeError(f"label must be bytes, got {type(label).__name__}")
        label = bytes(label)
        subkeys = []
        for i in range(n):
            digest = hashlib.sha256(self.secret_digest + label + i.to_bytes(4, "big")).digest()
            subkeys.append(int.from_bytes(digest[:4], "big"))
        return tuple(subkeys)


def _derive_hash_key(secret_bytes: bytes) -> int:
    digest = hashlib.sha256(secret_bytes).digest()
    return int.from_bytes(digest[:8], "big")


def _derive_secret_digest(secret_bytes: bytes) -> bytes:
    return hashlib.sha256(secret_bytes).digest()


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
    return WatermarkKey(
        key_id=key_id,
        hash_key=_derive_hash_key(secret_bytes),
        secret_digest=_derive_secret_digest(secret_bytes),
    )


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
                    "malformed WATERMARK_KEYS entry; expected 'key_id:hexsecret'"
                )
            key_id, hex_secret = entry.split(":", 1)
            key_id = key_id.strip()
            if not key_id:
                raise ValueError("malformed WATERMARK_KEYS entry; empty key_id")
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
