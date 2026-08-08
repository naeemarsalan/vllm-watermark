#!/usr/bin/env python3
"""Score one or more labeled corpora against the KGW detector and report
z-score distributions, TPR/FPR at a z-threshold, and per-length-bucket
breakdowns.

RUNS LOCALLY ONLY (needs a real HF tokenizer + the vllm_watermark
detector; transformers/torch are not installed in the bench pod -- see
bench_serving.py / gen_corpus.py, which are stdlib+requests only).

Importing vllm_watermark
------------------------
This script depends on Task A's vllm-free detector module:
    vllm_watermark.kgw.core     (KGWConfig)
    vllm_watermark.kgw.detector (detect_text, DEFAULT_Z_THRESHOLD)
    vllm_watermark.keys         (load_key)
It first tries a plain `import vllm_watermark` (works if the package was
installed, e.g. `pip install -e ".[detector]"` from the repo root); if
that fails, it falls back to inserting <repo-root>/src onto sys.path
(works from a bare checkout with no install step -- <repo-root> is this
file's grandparent directory, since this file lives at
<repo-root>/benchmarks/analyze_detection.py). Whichever path was used is
printed once at startup. If vllm_watermark cannot be imported by either
route, this script exits with a clear error rather than silently doing
nothing -- there is no other detector implementation to fall back to.

Corpus format
-------------
Each --corpus argument is `LABEL=PATH` (or bare `PATH`, in which case the
label defaults to the file's stem), pointing at a JSONL file of rows with
at least a `"text"` field:
  - benchmarks/gen_corpus.py rows: {prompt, text, finish_reason,
    completion_tokens, watermark: "on"|"off", key_id, model, temperature,
    request_ms}
  - benchmarks/fetch_human_corpus.py rows: {text, source, tokens}

Ground truth for TPR/FPR is read per-row from the `"watermark"` field
when present ("on" -> positive class, "off" -> negative class); rows with
no `"watermark"` field at all (e.g. the human corpus) are treated as
negative-class (unwatermarked) for FPR purposes -- this is what the task
spec means by "FPR at z>=4 for unwatermarked+human". `LABEL` is purely a
display tag for the report; it does not itself determine ground truth,
so a single corpus file with mixed on/off rows is handled correctly too.

Usage:
    python3 benchmarks/analyze_detection.py \\
        --corpus watermarked=benchmarks/data/corpus_watermarked.jsonl \\
        --corpus unwatermarked=benchmarks/data/corpus_unwatermarked.jsonl \\
        --corpus human=benchmarks/data/human_corpus.jsonl \\
        --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct \\
        --key-id default --z-threshold 4.0 \\
        --out benchmarks/data/report

Writes <out>.md and <out>.json (the extension, if any, on --out is
stripped and both are (re)written under that stem; default stem
"report").

The watermark key is loaded via vllm_watermark.keys.load_key(), i.e. from
WATERMARK_KEYS / WATERMARK_KEY(+WATERMARK_KEY_ID) env vars -- the same
key material the vLLM plugin used at generation time. Never passed on the
command line, never printed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _ensure_vllm_watermark_importable() -> str:
    """Make `vllm_watermark` importable; see module docstring "Importing
    vllm_watermark". Returns a short human-readable description of which
    route succeeded, for the startup banner."""
    try:
        import vllm_watermark

        return f"already importable ({vllm_watermark.__file__})"
    except ImportError:
        pass

    src_dir = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_dir))
    try:
        import vllm_watermark

        return f"sys.path fallback -> {src_dir} ({vllm_watermark.__file__})"
    except ImportError as exc:
        print(
            f"error: could not import vllm_watermark, even after adding {src_dir} "
            f"to sys.path: {exc}\n"
            "Either run `pip install -e .` (or `-e .[detector]`) from the repo "
            "root first, or run this script from a checkout where "
            "<repo-root>/src/vllm_watermark exists.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


_IMPORT_NOTE = _ensure_vllm_watermark_importable()

from vllm_watermark.keys import load_key  # noqa: E402
from vllm_watermark.kgw.core import KGWConfig  # noqa: E402
from vllm_watermark.kgw.detector import DEFAULT_Z_THRESHOLD, detect_text  # noqa: E402

DEFAULT_MODEL_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
LENGTH_BUCKETS = [
    ("<100", lambda n: n < 100),
    ("100-200", lambda n: 100 <= n < 200),
    ("200+", lambda n: n >= 200),
]


def parse_corpus_arg(value: str) -> "tuple[str, str]":
    if "=" in value:
        label, path = value.split("=", 1)
        label, path = label.strip(), path.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"empty label in --corpus {value!r}; expected LABEL=PATH")
        if not path:
            raise argparse.ArgumentTypeError(f"empty path in --corpus {value!r}; expected LABEL=PATH")
        return label, path
    return Path(value).stem, value


def load_jsonl_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  warning: {path}:{line_num}: malformed JSON, skipping row: {exc}", file=sys.stderr)
    return rows


def histogram(values: list[float], bins: int = 20) -> "tuple[list[int], list[float]]":
    """Fixed-bin-count histogram. Returns (counts, edges) where edges has
    bins+1 entries. Degenerate all-equal input gets a synthetic +/-0.5
    span so it still renders as one populated bin instead of dividing by
    zero."""
    if not values:
        return [0] * bins, [0.0] * (bins + 1)
    lo, hi = min(values), max(values)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width)
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += 1
    edges = [lo + i * width for i in range(bins + 1)]
    return counts, edges


def render_histogram_md(counts: list[int], edges: list[float], max_bar_width: int = 40) -> str:
    peak = max(counts) if counts else 0
    lines = []
    for i, count in enumerate(counts):
        bar_len = 0 if peak == 0 else round((count / peak) * max_bar_width)
        bar = "#" * bar_len
        lines.append(f"[{edges[i]:7.2f}, {edges[i + 1]:7.2f})  {bar:<{max_bar_width}}  {count}")
    return "\n".join(lines)


def rate(subset: list[dict]) -> "float | None":
    if not subset:
        return None
    return sum(1 for r in subset if r["prediction"]) / len(subset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable; a labeled JSONL corpus file (see module docstring 'Corpus format')",
    )
    parser.add_argument("--model-tokenizer", default=DEFAULT_MODEL_TOKENIZER)
    parser.add_argument("--key-id", default=None, help="watermark key id (default: WATERMARK_KEY_ID env or 'default')")
    parser.add_argument("--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=1,
        help="minimum num_tokens_scored (post-dedup pair count) for a row to be included in stats",
    )
    parser.add_argument("--gamma", type=float, default=0.25, help="must match the gamma used at generation time")
    parser.add_argument(
        "--out",
        default="report",
        help="output path stem; writes <stem>.md and <stem>.json (any .md/.json suffix given here is stripped)",
    )
    return parser


def _strip_known_suffix(out: str) -> str:
    for ext in (".md", ".json"):
        if out.endswith(ext):
            return out[: -len(ext)]
    return out


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"vllm_watermark import: {_IMPORT_NOTE}")

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        print(
            f"error: this script requires transformers (local-only tool): {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        key = load_key(key_id=args.key_id)
    except (RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Loading tokenizer/config {args.model_tokenizer!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)
    vocab_size = AutoConfig.from_pretrained(args.model_tokenizer).vocab_size
    cfg = KGWConfig(vocab_size=vocab_size, hash_key=key.hash_key, gamma=args.gamma)
    print(f"vocab_size={vocab_size} gamma={args.gamma} key_id={key.key_id} z_threshold={args.z_threshold}")

    try:
        corpus_specs = [parse_corpus_arg(c) for c in args.corpus]
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    corpora_report: dict[str, dict] = {}

    for label, path in corpus_specs:
        print(f"Scoring corpus {label!r} <- {path}")
        raw_rows = load_jsonl_rows(path)

        included: list[dict] = []
        skipped_no_text = 0
        skipped_too_short = 0
        excluded_min_tokens = 0

        for row in raw_rows:
            text = row.get("text")
            if not text or not isinstance(text, str):
                skipped_no_text += 1
                continue

            watermark_field = row.get("watermark")
            ground_truth = watermark_field if watermark_field in ("on", "off") else "off"
            # Rows with no "watermark" field at all (e.g. the human corpus,
            # whose rows only have text/source/tokens) are treated as the
            # negative class for FPR -- see module docstring.

            try:
                result = detect_text(
                    text, tokenizer, cfg, ignore_repeated_ngrams=True, z_threshold=args.z_threshold
                )
            except ValueError:
                skipped_too_short += 1
                continue

            if result.num_tokens_scored < args.min_tokens:
                excluded_min_tokens += 1
                continue

            included.append(
                {
                    "z_score": result.z_score,
                    "p_value": result.p_value,
                    "num_tokens_scored": result.num_tokens_scored,
                    "num_green": result.num_green,
                    "prediction": result.prediction,
                    "ground_truth": ground_truth,
                }
            )

        z_values = [r["z_score"] for r in included]
        positive_subset = [r for r in included if r["ground_truth"] == "on"]
        negative_subset = [r for r in included if r["ground_truth"] == "off"]

        length_bucket_stats = {}
        for bucket_name, predicate in LENGTH_BUCKETS:
            bucket_rows = [r for r in included if predicate(r["num_tokens_scored"])]
            bucket_pos = [r for r in bucket_rows if r["ground_truth"] == "on"]
            bucket_neg = [r for r in bucket_rows if r["ground_truth"] == "off"]
            length_bucket_stats[bucket_name] = {
                "n": len(bucket_rows),
                "mean_z": statistics.mean([r["z_score"] for r in bucket_rows]) if bucket_rows else None,
                "tpr": rate(bucket_pos),
                "fpr": rate(bucket_neg),
            }

        counts, edges = histogram(z_values, bins=20)

        corpora_report[label] = {
            "path": path,
            "n": len(included),
            "n_positive_ground_truth": len(positive_subset),
            "n_negative_ground_truth": len(negative_subset),
            "skipped_no_text": skipped_no_text,
            "skipped_too_short_to_score": skipped_too_short,
            "excluded_below_min_tokens": excluded_min_tokens,
            "mean_z": statistics.mean(z_values) if z_values else None,
            "median_z": statistics.median(z_values) if z_values else None,
            "min_z": min(z_values) if z_values else None,
            "max_z": max(z_values) if z_values else None,
            "tpr": rate(positive_subset),
            "fpr": rate(negative_subset),
            "length_buckets": length_bucket_stats,
            "histogram": {"counts": counts, "edges": edges},
            "rows": included,
        }

        n_str = len(included)
        print(
            f"  n={n_str} (skipped_no_text={skipped_no_text} "
            f"skipped_too_short={skipped_too_short} excluded_min_tokens={excluded_min_tokens})"
        )

    out_stem = _strip_known_suffix(args.out)
    md_path = f"{out_stem}.md"
    json_path = f"{out_stem}.json"

    config_summary = {
        "model_tokenizer": args.model_tokenizer,
        "vocab_size": vocab_size,
        "gamma": args.gamma,
        "key_id": key.key_id,
        "z_threshold": args.z_threshold,
        "min_tokens": args.min_tokens,
        "corpora": [{"label": label, "path": path} for label, path in corpus_specs],
    }

    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": config_summary, "corpora": corpora_report}, f, indent=2)

    md_lines = ["# KGW detection report", ""]
    md_lines.append(
        f"model_tokenizer=`{args.model_tokenizer}` vocab_size={vocab_size} gamma={args.gamma} "
        f"key_id=`{key.key_id}` z_threshold={args.z_threshold} min_tokens={args.min_tokens}"
    )
    md_lines.append("")
    md_lines.append("| corpus | n | mean z | median z | min z | max z | TPR (z>=thr, watermark=on) | FPR (z>=thr, watermark=off/human) |")
    md_lines.append("|---|---|---|---|---|---|---|---|")

    def _fmt(x, digits=3):
        return "n/a" if x is None else f"{x:.{digits}f}"

    for label, stats in corpora_report.items():
        md_lines.append(
            f"| {label} | {stats['n']} | {_fmt(stats['mean_z'])} | {_fmt(stats['median_z'])} | "
            f"{_fmt(stats['min_z'])} | {_fmt(stats['max_z'])} | {_fmt(stats['tpr'])} | {_fmt(stats['fpr'])} |"
        )

    md_lines.append("")

    for label, stats in corpora_report.items():
        md_lines.append(f"## {label}")
        md_lines.append("")
        md_lines.append(f"path: `{stats['path']}`")
        md_lines.append(
            f"n={stats['n']} (skipped_no_text={stats['skipped_no_text']} "
            f"skipped_too_short_to_score={stats['skipped_too_short_to_score']} "
            f"excluded_below_min_tokens={stats['excluded_below_min_tokens']}); "
            f"n_positive_ground_truth(watermark=on)={stats['n_positive_ground_truth']} "
            f"n_negative_ground_truth(watermark=off/human)={stats['n_negative_ground_truth']}"
        )
        md_lines.append("")
        md_lines.append("### Per-length-bucket (scored tokens)")
        md_lines.append("")
        md_lines.append("| bucket | n | mean z | TPR | FPR |")
        md_lines.append("|---|---|---|---|---|")
        for bucket_name, _ in LENGTH_BUCKETS:
            b = stats["length_buckets"][bucket_name]
            md_lines.append(f"| {bucket_name} | {b['n']} | {_fmt(b['mean_z'])} | {_fmt(b['tpr'])} | {_fmt(b['fpr'])} |")
        md_lines.append("")
        md_lines.append("### z-score histogram (20 bins)")
        md_lines.append("")
        md_lines.append("```")
        md_lines.append(render_histogram_md(stats["histogram"]["counts"], stats["histogram"]["edges"]))
        md_lines.append("```")
        md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
