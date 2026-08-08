#!/usr/bin/env python3
"""Phase 3 raw-evidence capture (committed so the EXPERIMENTS.md addendum's
raw transcript is reproducible — audit finding #5).

Run INSIDE the cluster bench pod:
    oc -n watermark cp benchmarks/phase3_raw_capture.py bench:/tmp/raw_capture.py
    oc -n watermark cp <samples json> bench:/tmp/e2e_samples.json
    oc -n watermark exec bench -- python3 /tmp/raw_capture.py
Samples file: {"kgw": [text,...], "synthid": [...], "clean": [...], "human": [...]}
(built from the gitignored benchmarks/data corpora; see EXPERIMENTS.md).

Prints raw HTTP request/response pairs; the ONLY alteration is eliding echoed
text content (marked [TEXT ELIDED, sha256:<prefix>]) so no submitted content
enters logs/transcripts."""
import argparse, json, hashlib, os, sys, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default="/tmp/e2e_samples.json")
ap.add_argument("--detector", default=os.environ.get("DETECTOR_URL", "http://detector:8000"))
ap.add_argument("--orchestrator", default=os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8033"))
ap.add_argument("--orchestrator-health", default=os.environ.get("ORCHESTRATOR_HEALTH_URL", "http://orchestrator:8034"))
ap.add_argument("--key-id", default=os.environ.get("VLLM_WATERMARK_KEY_ID"))
args = ap.parse_args()
if not args.key_id:
    sys.exit("error: watermark key id required — pass --key-id or set VLLM_WATERMARK_KEY_ID "
             "(sourced from your watermark-key Secret; never hardcoded here)")
samples = json.load(open(args.samples))
def redact(obj):
    """Elide any long string ANYWHERE in the structure (not only text/content
    keys) — error bodies and unexpected fields can echo submitted content too.
    Threshold 80 chars for known content keys, 200 for arbitrary strings."""
    if isinstance(obj, dict):
        return {k: (f"[TEXT ELIDED, sha256:{hashlib.sha256(v.encode()).hexdigest()[:16]}]"
                    if k in ("text", "content") and isinstance(v, str) and len(v) > 80 else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, str) and len(obj) > 200:
        return f"[STRING ELIDED len={len(obj)}, sha256:{hashlib.sha256(obj.encode()).hexdigest()[:16]}]"
    return obj
def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
print("### Raw orchestrator verdict matrix (server-side scheme authority; no client scheme params)")
for label in ("kgw", "synthid", "clean", "human"):
    for det in ("watermark-kgw", "watermark-synthid"):
        body = {"content": samples[label][1], "detectors": {det: {"key_id": args.key_id}}}
        st, r = post("http://orchestrator:8033/api/v2/text/detection/content", body)
        print(f"\n$ POST /api/v2/text/detection/content  content=<{label} sample 1> detectors={det}")
        print(f"HTTP {st} -> {json.dumps(redact(r), sort_keys=True)}")
print("\n### Raw signed direct-endpoint response (kgw sample 1)")
st, r = post("http://detector:8000/v1/watermark/detect",
             {"text": samples["kgw"][1], "scheme": "kgw", "key_id": args.key_id})
print(f"HTTP {st} -> {json.dumps(redact(r), sort_keys=True)}")
print("\n### Orchestrator health (dedicated port 8034)")
for path in ("/health", "/info"):
    with urllib.request.urlopen(f"{ORCH_HEALTH}{path}", timeout=15) as resp:
        print(f"GET {ORCH_HEALTH}{path} -> {resp.status} {resp.read().decode()[:200]}")
