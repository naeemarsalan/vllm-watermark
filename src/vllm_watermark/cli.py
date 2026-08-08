"""vllm_watermark CLI.

Usage:
    python -m vllm_watermark.cli detect --model-tokenizer <hf-id> \\
        --key-id <id> [--z-threshold 4.0] [--json] [--file text.txt]
    # or: ... | python -m vllm_watermark.cli detect --model-tokenizer <hf-id>

Reads text from --file if given, else stdin. Loads watermark key(s) from
env (see keys.py: WATERMARK_KEYS or WATERMARK_KEY/WATERMARK_KEY_ID) --
never from a CLI flag, and never prints key material.

Prints a JSON object to stdout:
    {"z_score": ..., "p_value": ..., "num_tokens_scored": ...,
     "num_green": ..., "prediction": ..., "key_id": ...}
--json selects compact single-line JSON; the default is pretty-printed
(indent=2) JSON -- both are valid JSON, this flag only changes formatting.

vocab_size: derived from `--model-tokenizer`'s AutoConfig.vocab_size (the
MODEL config, not the tokenizer's own length -- see core.py module
docstring for why that distinction matters) unless overridden with
--vocab-size.
"""

from __future__ import annotations

import argparse
import json
import sys

from vllm_watermark.keys import load_key


def _read_input_text(file_path: "str | None") -> str:
    if file_path:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _cmd_detect(args: argparse.Namespace) -> int:
    # Lazy import: transformers is optional for the package as a whole:
    # only the `detect` CLI command (which needs a real tokenizer) requires
    # it, so a plain `import vllm_watermark.cli` stays transformers-free.
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        print(
            "error: the 'detect' command requires transformers to be installed "
            f"(pip install transformers): {exc}",
            file=sys.stderr,
        )
        return 2

    from vllm_watermark.kgw.core import KGWConfig
    from vllm_watermark.kgw.detector import detect_text

    try:
        key = load_key(key_id=args.key_id)
    except (RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
        vocab_size = AutoConfig.from_pretrained(args.model_tokenizer).vocab_size

    cfg = KGWConfig(vocab_size=vocab_size, hash_key=key.hash_key, gamma=args.gamma)

    text = _read_input_text(args.file)
    result = detect_text(
        text,
        tokenizer,
        cfg,
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
        z_threshold=args.z_threshold,
    )

    payload = {
        "z_score": result.z_score,
        "p_value": result.p_value,
        "num_tokens_scored": result.num_tokens_scored,
        "num_green": result.num_green,
        "prediction": result.prediction,
        "key_id": key.key_id,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vllm_watermark.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Detect KGW watermark in text")
    detect_parser.add_argument(
        "--model-tokenizer",
        required=True,
        help="HF hub id or local path of the tokenizer (and, unless "
        "--vocab-size is given, the model config) used at generation time",
    )
    detect_parser.add_argument(
        "--key-id",
        default=None,
        help="Watermark key id to use (default: WATERMARK_KEY_ID env var, or 'default')",
    )
    detect_parser.add_argument("--z-threshold", type=float, default=4.0)
    detect_parser.add_argument("--gamma", type=float, default=0.25)
    detect_parser.add_argument(
        "--ignore-repeated-ngrams",
        action="store_true",
        help="Count each distinct (prev_token, target_token) pair at most once",
    )
    detect_parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override the vocab_size inferred from --model-tokenizer's model config",
    )
    detect_parser.add_argument("--file", default=None, help="Read text from this file instead of stdin")
    detect_parser.add_argument("--json", action="store_true", help="Compact single-line JSON output")
    detect_parser.set_defaults(func=_cmd_detect)

    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
