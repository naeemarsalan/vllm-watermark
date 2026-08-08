#!/usr/bin/env python3
"""Phase 2 quality spot-check (exploratory fluency proxy — see limitations).

Two measures over corpus completions, base model Qwen2.5-0.5B (float32, CPU):
1. UNCONDITIONAL completion PPL: exp(mean NLL of the completion tokens alone,
   no prompt context). NOT a generation-quality measure by itself — completions
   are out of context, and a watermark's whole job is shifting the sampling
   distribution, which this partially re-measures.
2. PROMPT-CONDITIONED completion PPL: exp(mean NLL of completion tokens given
   the row's own prompt; prompt token labels masked to -100). Closer to "how
   surprising was this completion where it actually appeared".

Selection: the FIRST --n rows of each corpus file. For the Phase 2 corpora this
is a PAIRED-PROMPT selection — the script VERIFIES it by hashing the prompt
lists across corpora and reports the result (byte-identical prompts confirmed
by SHA-256 for the recorded run). Note the distinction: selection is paired,
but the reported statistics are UNPAIRED per-corpus aggregate means (no
per-prompt differencing) — a paired estimator would have lower variance and is
left as an upgrade. Limitations: single small model, small n, PPL under the
generating model is a proxy — no human quality rating is implied. Usage:
  PYTHONPATH=src python3 benchmarks/quality_spotcheck.py \
      --corpus name=path [--corpus ...] [--n 15] [--max-tokens 256]
"""
import argparse, json, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--verify-pairing-only", action="store_true",
                    help="only run the prompt-set SHA-256 pairing check (no model load)")
    args = ap.parse_args()
    if args.n <= 0:
        ap.error(f"--n must be positive, got {args.n}")
    if args.max_tokens <= 0:
        ap.error(f"--max-tokens must be positive, got {args.max_tokens}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = None
    if not args.verify_pairing_only:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        model.eval()

    def nll(prompt, completion, conditioned):
        c_ids = tok.encode(completion, add_special_tokens=False)[: args.max_tokens]
        if len(c_ids) < 32:
            return None
        if conditioned and prompt:
            p_ids = tok.encode(prompt, add_special_tokens=False)
            ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + list(c_ids)
        else:
            ids, labels = list(c_ids), list(c_ids)
        t = torch.tensor([ids]); l = torch.tensor([labels])
        with torch.no_grad():
            return model(t, labels=l).loss.item()

    def distinct_n(texts, n):
        grams, total = set(), 0
        for tx in texts:
            ws = tx.split()
            gs = [tuple(ws[i:i+n]) for i in range(len(ws)-n+1)]
            grams.update(gs); total += len(gs)
        return len(grams)/total if total else float("nan")

    import hashlib, sys
    prompt_hashes = {}
    for spec in args.corpus:
        name, path = spec.split("=", 1)
        rows_h = [json.loads(l) for l in open(path)][: args.n]
        if len(rows_h) < args.n:
            sys.exit(f"error: corpus {name!r} has only {len(rows_h)} rows, need --n={args.n}")
        prompts = [r.get("prompt") for r in rows_h]
        if any(p is None for p in prompts):
            sys.exit(f"error: corpus {name!r} has rows without a 'prompt' field — "
                     "the pairing claim would be meaningless (hash over missing prompts refused)")
        prompt_hashes[name] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:16]
    vals = set(prompt_hashes.values())
    print(f"prompt-set sha256/16 per corpus: {prompt_hashes} -> "
          f"{'PAIRED (byte-identical prompts)' if len(vals)==1 else 'NOT paired'}")

    if args.verify_pairing_only:
        vals = set(prompt_hashes.values())
        print("PAIRED" if len(vals) == 1 else "NOT PAIRED")
        return

    for spec in args.corpus:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path)][: args.n]
        # distinct-n over the SAME first-max_tokens completion slice the PPL
        # uses (decode the truncated ids), so every reported measure covers
        # an identical span (audit alignment fix).
        texts = [tok.decode(tok.encode(r["text"], add_special_tokens=False)[: args.max_tokens])
                 for r in rows]
        u = [x for x in (nll(None, r["text"], False) for r in rows) if x is not None]
        c = [x for x in (nll(r.get("prompt"), r["text"], True) for r in rows) if x is not None]
        if not u or not c:
            sys.exit(f"error: corpus {name!r} has no eligible samples with >=32 completion "
                     f"tokens (uncond eligible: {len(u)}, conditioned: {len(c)})")
        pu, pc = math.exp(sum(u)/len(u)), math.exp(sum(c)/len(c))
        print(f"{name:14s} n={len(u):2d} PPL uncond mean={pu:7.2f}  "
              f"prompt-conditioned mean={pc:7.2f}  "
              f"distinct-1={distinct_n(texts,1):.3f} distinct-2={distinct_n(texts,2):.3f}")

if __name__ == "__main__":
    main()
