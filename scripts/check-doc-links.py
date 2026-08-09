#!/usr/bin/env python3
"""Validate repository-document links without third-party dependencies."""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


REPOSITORY = "naeemarsalan/vllm-watermark"
GITHUB_BLOB_PREFIX = f"https://github.com/{REPOSITORY}/blob/"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
EXPLICIT_ID = re.compile(r"<a\s+(?:[^>]*?\s)?id=['\"]([^'\"]+)['\"][^>]*>", re.IGNORECASE)


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.external_assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value for name, value in attrs}
        element_id = values.get("id")
        if element_id:
            self.ids.add(element_id)
        href = values.get("href")
        if href:
            self.links.append(href)
        source = values.get("src")
        if source and tag in {"script", "img", "iframe", "audio", "video", "source"}:
            if urlsplit(source).scheme in {"http", "https"}:
                self.external_assets.append(source)
        if tag == "link" and href:
            rel = (values.get("rel") or "").lower().split()
            if "stylesheet" in rel and urlsplit(href).scheme in {"http", "https"}:
                self.external_assets.append(href)


def markdown_slug(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"[*_~]", "", text)
    text = text.replace(chr(96), "")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    # GitHub replaces each whitespace character after punctuation removal.
    # Two spaces around a removed em dash or plus sign therefore become "--".
    return re.sub(r"\s", "-", text.strip())


def document_ids(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = DocumentParser()
        parser.feed(source)
        return parser.ids

    ids = set(EXPLICIT_ID.findall(source))
    seen: dict[str, int] = {}
    for match in HEADING.finditer(source):
        base = markdown_slug(match.group(2))
        count = seen.get(base, 0)
        seen[base] = count + 1
        ids.add(base if count == 0 else f"{base}-{count}")
    return ids


def document_links(path: Path) -> tuple[list[str], list[str]]:
    source = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = DocumentParser()
        parser.feed(source)
        return parser.links, parser.external_assets
    return [html.unescape(match.group(1)) for match in MARKDOWN_LINK.finditer(source)], []


def github_blob_target(url: str, root: Path) -> tuple[Path, str] | None:
    if not url.startswith(GITHUB_BLOB_PREFIX):
        return None
    parsed = urlsplit(url)
    parts = unquote(parsed.path).lstrip("/").split("/")
    if len(parts) < 5 or parts[0:2] != REPOSITORY.split("/") or parts[2] != "blob":
        return None
    return root.joinpath(*parts[4:]), unquote(parsed.fragment)


def local_target(source: Path, href: str, root: Path) -> tuple[Path, str] | None:
    mapped = github_blob_target(href, root)
    if mapped:
        return mapped
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return None
    if not parsed.path:
        return source, unquote(parsed.fragment)
    target = (source.parent / unquote(parsed.path)).resolve()
    return target, unquote(parsed.fragment)


def check_external(url: str) -> str | None:
    clean_url = url.split("#", 1)[0]
    # Some documentation CDNs deny repository-specific crawler identifiers
    # while accepting a conventional link-checker agent. Keep the repository
    # URL so operators can identify the source of the request.
    headers = {
        "User-Agent": (
            "GitHub-Link-Checker/1.0 "
            "(+https://github.com/naeemarsalan/vllm-watermark)"
        )
    }
    for method in ("HEAD", "GET"):
        request = Request(clean_url, headers=headers, method=method)
        try:
            with urlopen(request, timeout=20) as response:
                if 200 <= response.status < 400:
                    return None
                return f"HTTP {response.status}"
        except HTTPError as error:
            if method == "HEAD" and error.code in {403, 405, 501}:
                continue
            return f"HTTP {error.code}"
        except (URLError, TimeoutError) as error:
            if method == "HEAD":
                continue
            return str(error.reason if isinstance(error, URLError) else error)
    return "request failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--external", action="store_true", help="also request external URLs")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    errors: list[str] = []
    external: set[str] = set()
    local_count = 0

    for supplied in args.documents:
        source = supplied.resolve()
        if not source.is_file():
            errors.append(f"{supplied}: document does not exist")
            continue
        links, assets = document_links(source)
        for asset in assets:
            errors.append(f"{supplied}: external page asset is not self-contained: {asset}")
        for href in links:
            if href.startswith(("mailto:", "tel:", "data:", "javascript:")):
                continue
            target_info = local_target(source, href, root)
            if target_info is None:
                external.add(href)
                continue
            target, fragment = target_info
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f"{supplied}: link escapes repository: {href}")
                continue
            local_count += 1
            if not target.exists():
                errors.append(f"{supplied}: missing target: {href}")
                continue
            if fragment and target.is_file():
                if fragment not in document_ids(target):
                    errors.append(f"{supplied}: missing fragment in {target.relative_to(root)}: #{fragment}")

    if args.external:
        for url in sorted(external):
            failure = check_external(url)
            if failure:
                errors.append(f"external link failed ({failure}): {url}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    mode = "requested" if args.external else "not requested"
    print(
        f"clean: {len(args.documents)} documents, {local_count} local references, "
        f"{len(external)} external URLs (network checks {mode})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
