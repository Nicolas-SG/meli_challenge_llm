"""Detect navigation-only Markdown files (sidebars, coverpages, link menus).

A file is considered navigation when it is structurally a menu rather than
documentation:

* it has at most ``max_headings`` headings (a coverpage title, or none at all),
* at least ``min_link_ratio`` of its non-empty lines contain a link or image,
* and the remaining prose (lines without links/images/headings) has at most
  ``max_prose_words`` words (e.g. a one-line tagline).

Thresholds were picked by looking at the per-file distribution in docs_raw:
every sidebar/coverpage has <= 1 heading, link ratio >= 0.5 and <= 21 prose
words, while short real pages (e.g. a diagram page) have several headings.
"""

from __future__ import annotations

import html
import re

_LINK_OR_IMAGE_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)|<(?:img|a)\b[^>]*>", re.I)
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]|^\s*<h[1-6]\b", re.I)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"\w+")


def navigation_stats(markdown: str) -> dict[str, float]:
    """Structural stats used to decide whether a file is navigation."""
    text = html.unescape(_HTML_COMMENT_RE.sub("", markdown))
    lines = [line for line in text.splitlines() if line.strip()]
    headings = [line for line in lines if _HEADING_RE.match(line)]
    link_lines = [line for line in lines if _LINK_OR_IMAGE_RE.search(line)]
    prose_lines = [
        line
        for line in lines
        if not _LINK_OR_IMAGE_RE.search(line) and not _HEADING_RE.match(line)
    ]
    prose_words = sum(
        len(_WORD_RE.findall(_HTML_TAG_RE.sub("", line))) for line in prose_lines
    )
    return {
        "n_lines": len(lines),
        "n_headings": len(headings),
        "link_ratio": len(link_lines) / len(lines) if lines else 0.0,
        "prose_words": prose_words,
    }


def is_navigation_file(
    markdown: str,
    max_headings: int = 1,
    min_link_ratio: float = 0.5,
    max_prose_words: int = 30,
) -> bool:
    stats = navigation_stats(markdown)
    return (
        stats["n_lines"] > 0
        and stats["n_headings"] <= max_headings
        and stats["link_ratio"] >= min_link_ratio
        and stats["prose_words"] <= max_prose_words
    )
