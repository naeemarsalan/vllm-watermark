#!/usr/bin/env python3
"""Build a human-text control corpus from public-domain Project Gutenberg
books, chunked to ~256 tokens with the target model's own tokenizer.

RUNS LOCALLY ONLY (needs network + a real HF tokenizer -- transformers is
NOT installed in the bench pod; see benchmarks/bench_serving.py and
gen_corpus.py, which are stdlib+requests only for that reason).

Books (stable, well-known Gutenberg IDs; plain-text mirrors verified
reachable 2026-08-08 via both URL templates below):
    1342  Pride and Prejudice        -- Jane Austen
    84    Frankenstein                -- Mary Shelley
    11    Alice's Adventures in Wonderland -- Lewis Carroll
    2701  Moby-Dick                   -- Herman Melville

For each book, two URL templates are tried in order (Gutenberg serves the
same plain-text file at both paths for most catalog entries; if the first
404s or errors, the second is tried; if both fail, that book is skipped
with a warning and the run continues with the remaining books -- "falls
back gracefully if a mirror fails" per the task spec):
    https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt
    https://www.gutenberg.org/files/{id}/{id}-0.txt

Gutenberg header/footer are stripped via the standard boilerplate
markers ("*** START OF ... ***" / "*** END OF ... ***"), which Gutenberg's
plain-text distribution format guarantees on the file's own line.

Chunking: the cleaned book text is tokenized once (whole-book, no special
tokens) with the HF tokenizer for --model-tokenizer, then sliced into
non-overlapping windows of --chunk-tokens token ids and decoded back to
text. Windows shorter than --chunk-tokens (the remainder at the end of a
book) are dropped rather than kept as partial/short chunks. Each decoded
chunk is then screened by _is_low_quality_chunk() to drop chunks that are
mostly whitespace, mostly non-alphabetic (e.g. a table of contents, a
chapter-number list), or otherwise not representative running prose.

--n chunks (default 150) are then sampled from the pooled, filtered chunk
set across all books with a fixed seed (default 42) via random.Random,
for reproducibility.

Usage:
    python3 benchmarks/fetch_human_corpus.py \\
        --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct \\
        --n 150 --seed 42 \\
        --out benchmarks/data/human_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys

import requests

BOOKS = [
    {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen"},
    {"id": 84, "title": "Frankenstein; or, The Modern Prometheus", "author": "Mary Shelley"},
    {"id": 11, "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll"},
    {"id": 2701, "title": "Moby-Dick; or, The Whale", "author": "Herman Melville"},
]

URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
]

_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)

DEFAULT_MODEL_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_N = 150
DEFAULT_SEED = 42
DEFAULT_CHUNK_TOKENS = 256
FETCH_TIMEOUT_S = 30.0


def fetch_book_text(book_id: int) -> "str | None":
    for template in URL_TEMPLATES:
        url = template.format(id=book_id)
        try:
            resp = requests.get(url, timeout=FETCH_TIMEOUT_S)
        except requests.RequestException as exc:
            print(f"  warning: {url} -> {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if resp.status_code != 200:
            print(f"  warning: {url} -> HTTP {resp.status_code}", file=sys.stderr)
            continue
        resp.encoding = resp.encoding or "utf-8"
        return resp.text
    return None


def strip_gutenberg_boilerplate(raw_text: str) -> "str | None":
    start_match = _START_RE.search(raw_text)
    end_match = _END_RE.search(raw_text)
    if not start_match or not end_match or end_match.start() <= start_match.end():
        return None
    return raw_text[start_match.end() : end_match.start()].strip()


def _is_low_quality_chunk(text: str) -> bool:
    """Heuristic screen for chunks that are mostly whitespace, tables, or
    other non-prose boilerplate (e.g. a table of contents, an all-caps
    chapter-heading run, a list of illustrations)."""
    stripped = text.strip()
    if len(stripped) < 100:
        return True
    non_space = [c for c in stripped if not c.isspace()]
    if not non_space:
        return True
    alpha_ratio = sum(1 for c in non_space if c.isalpha()) / len(non_space)
    if alpha_ratio < 0.6:
        return True
    # Many short lines packed together (tables, verse-like lists of
    # numbers/headings) show up as an unusually high newline density
    # relative to prose, which wraps as long lines.
    if stripped.count("\n") / len(stripped) > 0.03:
        return True
    return False


def chunk_book(
    book_text: str, tokenizer, chunk_tokens: int
) -> list[list[int]]:
    token_ids = tokenizer.encode(book_text, add_special_tokens=False)
    windows = []
    for start in range(0, len(token_ids) - chunk_tokens + 1, chunk_tokens):
        windows.append(token_ids[start : start + chunk_tokens])
    return windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tokenizer", default=DEFAULT_MODEL_TOKENIZER)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--out", default="benchmarks/data/human_corpus.jsonl")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(
            f"error: this script requires transformers (local-only tool): {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Loading tokenizer {args.model_tokenizer!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)

    all_chunks: list[dict] = []  # {"text": str, "source": str, "tokens": int}
    for book in BOOKS:
        print(f"Fetching Gutenberg #{book['id']}: {book['title']!r} ...")
        raw_text = fetch_book_text(book["id"])
        if raw_text is None:
            print(f"  warning: all mirrors failed for #{book['id']}; skipping this book", file=sys.stderr)
            continue

        cleaned = strip_gutenberg_boilerplate(raw_text)
        if cleaned is None:
            print(
                f"  warning: could not find START/END Gutenberg markers for #{book['id']}; skipping this book",
                file=sys.stderr,
            )
            continue

        source_label = f"gutenberg:{book['id']}:{book['title']}"
        windows = chunk_book(cleaned, tokenizer, args.chunk_tokens)

        kept = 0
        dropped = 0
        for window_ids in windows:
            text = tokenizer.decode(window_ids, skip_special_tokens=True)
            if _is_low_quality_chunk(text):
                dropped += 1
                continue
            all_chunks.append({"text": text, "source": source_label, "tokens": len(window_ids)})
            kept += 1
        print(f"  {len(windows)} candidate {args.chunk_tokens}-token windows: kept {kept}, dropped {dropped}")

    if not all_chunks:
        print("error: no chunks produced from any book -- nothing to sample", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    if args.n >= len(all_chunks):
        sample = list(all_chunks)
        print(
            f"warning: requested --n {args.n} >= {len(all_chunks)} available filtered chunks; "
            "using all of them",
            file=sys.stderr,
        )
    else:
        sample = rng.sample(all_chunks, args.n)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for row in sample:
            out_f.write(json.dumps(row) + "\n")

    token_counts = [row["tokens"] for row in sample]
    by_source: dict[str, int] = {}
    for row in sample:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1

    print()
    print("=== Summary ===")
    print(f"total filtered chunks available across all books: {len(all_chunks)}")
    print(f"sampled (seed={args.seed}): {len(sample)}")
    print(f"tokens per chunk: min={min(token_counts)} max={max(token_counts)} (all == --chunk-tokens by construction)")
    print("chunks per source:")
    for source, count in sorted(by_source.items()):
        print(f"  {count:4d}  {source}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
