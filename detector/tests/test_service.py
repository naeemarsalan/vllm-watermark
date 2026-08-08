# SPDX-License-Identifier: Apache-2.0
"""Local-only test suite for `detector/app.py` (Task B3).

Run with:
    /usr/bin/python3 -m pytest detector/tests/test_service.py -v
(needs detector/tests/conftest.py's sys.path bootstrap, which runs
automatically -- see that file's docstring; no editable install needed.)

Uses `MODEL_TOKENIZER=gpt2` (cached locally -- see AGENTS.md environment
notes: "Local is fine for detector math, unit tests, docs") with
`WATERMARK_VOCAB_SIZE=50257` (gpt2's own `tok.vocab_size`, so no
tokenizer/model vocab-padding mismatch complicates these tests -- see
app.py module docstring "WATERMARK_VOCAB_SIZE" for why that distinction
matters on a real deployment). Self-consistency is what's under test here
(generation-side greenlist math and detection-side greenlist math using the
identical `vllm_watermark.kgw.core` functions and the identical dummy key),
not agreement with the Qwen production model.

Covers the four Task B3 test points:
    (a) TestContentsEndpoint -- synthetic watermarked stream (built with
        vllm_watermark.kgw.core greenlist bias, decoded via gpt2) detected
        via /api/v1/text/contents; clean text -> empty list.
    (b) TestDetectEndpoint / TestSigning -- direct endpoint coherent
        fields + a throwaway-PEM-backed valid detached JWS.
    (c) TestZeroRetention -- content never appears in captured logs.
    (d) malformed body -> 422 (spread across TestContentsEndpoint /
        TestDetectEndpoint, one assertion per endpoint's own request shape).
"""

from __future__ import annotations

import hashlib
import json
import logging

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import app as detector_app
from vllm_watermark.kgw.core import KGWConfig, greenlist_ids

DUMMY_HEX_SECRET = "aa" * 16  # obviously-dummy test secret, AGENTS.md #3
GPT2_VOCAB_SIZE = 50257

CLEAN_TEXT = (
    "The quick brown fox jumps over the lazy dog near the riverbank every "
    "single morning before sunrise, watching the world wake up slowly "
    "around it while the birds begin to sing in the tall oak trees."
)


def _make_kgw_watermarked_text(app_state, num_tokens: int = 200) -> str:
    """Build a synthetic token stream that is, by construction, 100% inside
    the KGW greenlist at every scored position -- using
    `vllm_watermark.kgw.core.greenlist_ids` directly (the same function the
    detector-under-test's own scoring path uses), then decode it through
    the SAME tokenizer the service loaded. Only "new word, alphabetic" gpt2
    BPE pieces (surface form starting with "Ġ", i.e. a leading space,
    with an alphabetic remainder) are picked from each greenlist -- verified
    by execution at a Python prompt before this test was written (a plain
    all-greenlist construction, without this filter, does NOT reliably
    round-trip through gpt2's BPE decode -> re-encode) that this keeps the
    decode -> re-tokenize round trip byte-identical to the token ids picked
    here, so the service (which re-tokenizes the submitted text itself, not
    these ids) sees the exact greenlist membership this function
    constructed, not a shifted/reinterpreted one. The `assert` a few lines
    below is that same claim, pinned as an executable, re-checked-on-every-run
    guard rather than a one-off manual observation.
    """
    tok = app_state.tokenizer
    key = app_state.default_key
    cfg = KGWConfig(vocab_size=app_state.vocab_size, hash_key=key.hash_key, gamma=0.25, delta=2.0)

    token_ids = tok.encode(" The", add_special_tokens=False)
    for _ in range(num_tokens):
        prev = token_ids[-1]
        greenlist = greenlist_ids(prev, cfg).tolist()
        candidates = [
            t
            for t in greenlist
            if (piece := tok.convert_ids_to_tokens([t])[0]).startswith("Ġ") and piece[1:].isalpha()
        ]
        token_ids.append(candidates[0] if candidates else greenlist[0])

    text = tok.decode(token_ids)
    # Guard the round-trip assumption the docstring above claims, so a
    # future tokenizer/version change fails loudly here instead of quietly
    # producing a flaky detection assertion below.
    assert tok.encode(text, add_special_tokens=False) == token_ids
    return text


@pytest.fixture
def base_env(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", DUMMY_HEX_SECRET)
    monkeypatch.setenv("WATERMARK_KEY_ID", "test-key")
    monkeypatch.setenv("MODEL_TOKENIZER", "gpt2")
    monkeypatch.setenv("WATERMARK_VOCAB_SIZE", str(GPT2_VOCAB_SIZE))
    monkeypatch.delenv("WATERMARK_KEYS", raising=False)
    monkeypatch.delenv("SIGNING_KEY_PATH", raising=False)
    monkeypatch.delenv("SIGNING_KEY_ID", raising=False)
    monkeypatch.delenv("WATERMARK_DETECTOR_SCHEME", raising=False)
    yield monkeypatch


@pytest.fixture
def client(base_env):
    application = detector_app.create_app()
    with TestClient(application) as c:
        yield c


# ---------------------------------------------------------------------------
# (a) /api/v1/text/contents -- TrustyAI-contract endpoint
# ---------------------------------------------------------------------------


class TestContentsEndpoint:
    def test_watermarked_text_detected_above_threshold(self, client):
        wm_text = _make_kgw_watermarked_text(client.app.state)
        resp = client.post("/api/v1/text/contents", json={"contents": [wm_text]})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1  # one inner list per submitted content, in order
        detections = body[0]
        assert len(detections) == 1

        d = detections[0]
        assert d["start"] == 0
        assert d["end"] == len(wm_text)
        assert d["text"] == wm_text
        assert d["detection_type"] == "watermark"
        assert d["detection"] == "kgw-watermark"
        assert 0.0 <= d["score"] <= 1.0
        assert d["score"] > 0.5, "calibrated score must clear the midpoint for a well-above-threshold z"
        assert d["metadata"]["z_score"] >= 4.0  # WATERMARK_Z_THRESHOLD default
        assert d["metadata"]["scheme"] == "kgw"
        assert d["metadata"]["key_id"] == "test-key"

    def test_clean_text_yields_empty_list(self, client):
        resp = client.post("/api/v1/text/contents", json={"contents": [CLEAN_TEXT]})
        assert resp.status_code == 200
        assert resp.json() == [[]]

    def test_mixed_batch_preserves_order(self, client):
        wm_text = _make_kgw_watermarked_text(client.app.state)
        resp = client.post("/api/v1/text/contents", json={"contents": [CLEAN_TEXT, wm_text, CLEAN_TEXT]})
        body = resp.json()
        assert len(body) == 3
        assert body[0] == []
        assert len(body[1]) == 1
        assert body[2] == []

    def test_alias_routes_force_scheme_over_detector_params(self, client):
        wm_text = _make_kgw_watermarked_text(client.app.state)

        # The /kgw alias forces scheme=kgw even though detector_params asks
        # for synthid -- see app.py "Scheme selection".
        r_kgw = client.post(
            "/kgw/api/v1/text/contents",
            json={"contents": [wm_text], "detector_params": {"scheme": "synthid"}},
        )
        assert r_kgw.status_code == 200
        kgw_detections = r_kgw.json()[0]
        assert len(kgw_detections) == 1
        assert kgw_detections[0]["metadata"]["scheme"] == "kgw"

        # The generic route, by contrast, DOES honor detector_params.scheme
        # -- scoring KGW-crafted text under SynthID math should not
        # false-positive.
        r_generic = client.post(
            "/api/v1/text/contents",
            json={"contents": [wm_text], "detector_params": {"scheme": "synthid"}},
        )
        assert r_generic.status_code == 200
        assert r_generic.json() == [[]]

    def test_detector_params_key_id_selects_key(self, client, base_env):
        base_env.setenv("WATERMARK_KEYS", f"test-key:{DUMMY_HEX_SECRET},other-key:{'bb' * 16}")
        base_env.delenv("WATERMARK_KEY", raising=False)
        application = detector_app.create_app()
        with TestClient(application) as c:
            resp = c.post(
                "/api/v1/text/contents",
                json={"contents": [CLEAN_TEXT], "detector_params": {"key_id": "other-key"}},
            )
        assert resp.status_code == 200

    def test_unknown_detector_params_key_id_400(self, client):
        resp = client.post(
            "/api/v1/text/contents",
            json={"contents": [CLEAN_TEXT], "detector_params": {"key_id": "does-not-exist"}},
        )
        assert resp.status_code == 400

    # --- (d) malformed body -> 422 ---
    def test_malformed_body_contents_wrong_type_422(self, client):
        resp = client.post("/api/v1/text/contents", json={"contents": "not-a-list"})
        assert resp.status_code == 422

    def test_malformed_body_missing_contents_422(self, client):
        resp = client.post("/api/v1/text/contents", json={})
        assert resp.status_code == 422

    def test_malformed_body_empty_contents_422(self, client):
        resp = client.post("/api/v1/text/contents", json={"contents": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# (b) /v1/watermark/detect -- direct endpoint
# ---------------------------------------------------------------------------

_REQUIRED_DETECT_FIELDS = (
    "scheme",
    "key_id",
    "verdict",
    "z_score",
    "p_value",
    "score",
    "num_tokens_scored",
    "detector_version",
    "model_tokenizer",
    "scheme_details",
    "signature",
    "signing",
)


class TestDetectEndpoint:
    def test_watermarked_text_coherent_fields(self, client):
        wm_text = _make_kgw_watermarked_text(client.app.state)
        resp = client.post("/v1/watermark/detect", json={"text": wm_text})
        assert resp.status_code == 200
        body = resp.json()

        for field in _REQUIRED_DETECT_FIELDS:
            assert field in body, f"missing field {field!r} in {body!r}"

        assert body["scheme"] == "kgw"
        assert body["key_id"] == "test-key"
        assert body["verdict"] is True
        assert body["z_score"] >= 4.0
        assert 0.0 <= body["p_value"] <= 1.0
        assert 0.0 <= body["score"] <= 1.0
        assert body["num_tokens_scored"] > 0
        assert body["model_tokenizer"] == "gpt2"
        assert body["detector_version"].startswith("vllm-watermark-detector/")
        assert body["scheme_details"]["gamma"] == pytest.approx(0.25)
        assert body["signature"] is None
        assert body["signing"] == "disabled"

    def test_clean_text_negative_verdict(self, client):
        resp = client.post("/v1/watermark/detect", json={"text": CLEAN_TEXT})
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] is False
        assert body["z_score"] < 4.0

    def test_explicit_scheme_override(self, client):
        resp = client.post("/v1/watermark/detect", json={"text": CLEAN_TEXT, "scheme": "synthid"})
        assert resp.status_code == 200
        assert resp.json()["scheme"] == "synthid"

    def test_batch_texts_results_list(self, client):
        wm_text = _make_kgw_watermarked_text(client.app.state)
        resp = client.post("/v1/watermark/detect", json={"texts": [wm_text, CLEAN_TEXT]})
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert len(body["results"]) == 2
        assert body["results"][0]["verdict"] is True
        assert body["results"][1]["verdict"] is False
        assert body["signing"] == "disabled"

    def test_unknown_key_id_400(self, client):
        resp = client.post("/v1/watermark/detect", json={"text": CLEAN_TEXT, "key_id": "does-not-exist"})
        assert resp.status_code == 400

    def test_no_keys_configured_503(self, monkeypatch):
        monkeypatch.setenv("MODEL_TOKENIZER", "gpt2")
        monkeypatch.setenv("WATERMARK_VOCAB_SIZE", str(GPT2_VOCAB_SIZE))
        monkeypatch.delenv("WATERMARK_KEY", raising=False)
        monkeypatch.delenv("WATERMARK_KEYS", raising=False)
        application = detector_app.create_app()
        with TestClient(application) as c:
            resp = c.post("/v1/watermark/detect", json={"text": CLEAN_TEXT})
        assert resp.status_code == 503

    # --- (d) malformed body -> 422 ---
    def test_malformed_body_neither_text_nor_texts_422(self, client):
        assert client.post("/v1/watermark/detect", json={"key_id": "test-key"}).status_code == 422

    def test_malformed_body_both_text_and_texts_422(self, client):
        assert client.post("/v1/watermark/detect", json={"text": "a", "texts": ["b"]}).status_code == 422

    def test_malformed_body_bad_scheme_422(self, client):
        resp = client.post("/v1/watermark/detect", json={"text": CLEAN_TEXT, "scheme": "not-a-real-scheme"})
        assert resp.status_code == 422

    def test_malformed_body_empty_texts_422(self, client):
        assert client.post("/v1/watermark/detect", json={"texts": []}).status_code == 422

    def test_malformed_body_wrong_type_422(self, client):
        assert client.post("/v1/watermark/detect", json={"text": 12345}).status_code == 422


# ---------------------------------------------------------------------------
# (b, continued) detached-JWS signing
# ---------------------------------------------------------------------------


class TestSigning:
    def test_unsigned_when_no_signing_key_configured(self, client):
        resp = client.post("/v1/watermark/detect", json={"text": CLEAN_TEXT})
        body = resp.json()
        assert body["signature"] is None
        assert body["signing"] == "disabled"

    def test_valid_detached_jws_with_throwaway_test_pem(self, base_env, tmp_path):
        # Obviously-throwaway RSA test key, generated fresh per test run --
        # never a real/production key (AGENTS.md #3).
        priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        key_path = tmp_path / "throwaway_test_signing_key.pem"
        key_path.write_bytes(pem)
        base_env.setenv("SIGNING_KEY_PATH", str(key_path))
        base_env.setenv("SIGNING_KEY_ID", "test-kid-1")

        application = detector_app.create_app()
        with TestClient(application) as c:
            resp = c.post("/v1/watermark/detect", json={"text": CLEAN_TEXT})
        assert resp.status_code == 200
        body = resp.json()
        assert body["signing"] == "enabled"
        assert isinstance(body["signature"], str) and body["signature"]

        signature = body.pop("signature")
        body.pop("signing")
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

        decoded = jwt.api_jws.decode_complete(
            signature, key=priv.public_key(), algorithms=["RS256"], detached_payload=canonical
        )
        assert decoded["header"]["alg"] == "RS256"
        assert decoded["header"]["kid"] == "test-kid-1"
        assert decoded["header"]["b64"] is False  # RFC 7797 detached-payload marker

        # A verifier without the matching payload cannot fabricate a valid check.
        with pytest.raises(jwt.exceptions.InvalidSignatureError):
            jwt.api_jws.decode_complete(
                signature, key=priv.public_key(), algorithms=["RS256"], detached_payload=canonical + b"tampered"
            )

    def test_batch_response_also_signed(self, base_env, tmp_path):
        priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        key_path = tmp_path / "throwaway_test_signing_key2.pem"
        key_path.write_bytes(pem)
        base_env.setenv("SIGNING_KEY_PATH", str(key_path))

        application = detector_app.create_app()
        with TestClient(application) as c:
            resp = c.post("/v1/watermark/detect", json={"texts": [CLEAN_TEXT]})
        body = resp.json()
        assert body["signing"] == "enabled"
        signature = body.pop("signature")
        body.pop("signing")
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        jwt.api_jws.decode_complete(signature, key=priv.public_key(), algorithms=["RS256"], detached_payload=canonical)


# ---------------------------------------------------------------------------
# (c) Zero retention
# ---------------------------------------------------------------------------


class TestZeroRetention:
    def test_content_never_appears_in_captured_logs(self, client, caplog):
        wm_text = _make_kgw_watermarked_text(client.app.state)

        with caplog.at_level(logging.DEBUG):
            client.post("/api/v1/text/contents", json={"contents": [wm_text, CLEAN_TEXT]})
            client.post("/v1/watermark/detect", json={"text": wm_text})
            client.post("/v1/watermark/detect", json={"texts": [wm_text, CLEAN_TEXT]})

        all_log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert wm_text not in all_log_text
        assert CLEAN_TEXT not in all_log_text
        # Partial-leakage guard, not just a whole-string match.
        assert wm_text[:40] not in all_log_text
        assert CLEAN_TEXT[:40] not in all_log_text

        # The one ALLOWED content-derived value (sha256[:16]) IS expected to
        # appear -- confirms the assertions above aren't vacuously true
        # because nothing was logged at all.
        wm_digest = hashlib.sha256(wm_text.encode("utf-8")).hexdigest()[:16]
        clean_digest = hashlib.sha256(CLEAN_TEXT.encode("utf-8")).hexdigest()[:16]
        assert wm_digest in all_log_text
        assert clean_digest in all_log_text

    def test_insufficient_tokens_error_never_embeds_text(self, client, caplog):
        # A single short token's worth of text is below KGW's 2-token floor
        # -- exercises the InsufficientTokensError path specifically.
        short_text = "Hi"
        with caplog.at_level(logging.DEBUG):
            resp = client.post("/v1/watermark/detect", json={"text": short_text})
        assert resp.status_code == 422
        assert short_text not in resp.text
        all_log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert short_text not in all_log_text

    def test_no_body_logging_middleware_installed(self, client):
        """Assert no request-body logging middleware -- see app.py module
        docstring 'Zero retention'. An empty middleware stack means nothing
        in this app could be dumping request bodies to a log/sink."""
        assert client.app.user_middleware == []


# ---------------------------------------------------------------------------
# GET /health, GET /ready
# ---------------------------------------------------------------------------


class TestHealthReady:
    def test_health_always_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ready_200_when_keys_and_tokenizer_loaded(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tokenizer_loaded"] is True
        assert body["key_ids"] == ["test-key"]

    def test_ready_503_when_no_keys_configured(self, monkeypatch):
        monkeypatch.setenv("MODEL_TOKENIZER", "gpt2")
        monkeypatch.setenv("WATERMARK_VOCAB_SIZE", str(GPT2_VOCAB_SIZE))
        monkeypatch.delenv("WATERMARK_KEY", raising=False)
        monkeypatch.delenv("WATERMARK_KEYS", raising=False)
        application = detector_app.create_app()
        with TestClient(application) as c:
            health_resp = c.get("/health")
            ready_resp = c.get("/ready")
        # /health stays healthy (liveness) even though /ready correctly
        # gates on configuration (readiness) -- see app.py module docstring.
        assert health_resp.status_code == 200
        assert ready_resp.status_code == 503
        assert ready_resp.json()["detail"]["keys_configured"] is False

    def test_vocab_size_fallback_warns_but_does_not_crash(self, base_env, caplog):
        base_env.delenv("WATERMARK_VOCAB_SIZE", raising=False)
        with caplog.at_level(logging.WARNING):
            application = detector_app.create_app()
            with TestClient(application) as c:
                resp = c.get("/ready")
        assert resp.status_code == 200
        assert application.state.vocab_size == GPT2_VOCAB_SIZE  # len(gpt2 tokenizer)
        assert any("WATERMARK_VOCAB_SIZE not set" in r.getMessage() for r in caplog.records)
