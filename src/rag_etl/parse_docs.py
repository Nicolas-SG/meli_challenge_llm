"""Parse a tree of Markdown docs into a section-level DataFrame.

Expected layout under the root directory:

    <root>/<project_name>/<version>/**/*.md

Pipeline (each step is a function so it can be run piece by piece):

    load_files -> drop_navigation_files -> explode_sections
               -> flag_todo -> flag_near_duplicates

Each Markdown file is split on headings that are outside fenced code blocks:
ATX headings (``#`` .. ``######``) and HTML headings (``<h1>`` .. ``<h6>``).
Every resulting section becomes one row:

    path            file path relative to <root> (provenance)
    file_name       file name, e.g. "README.md"
    project_name    first directory level under <root>
    version         second directory level under <root>
    section         heading breadcrumb, e.g. "Intro > Setup > Docker"
                    ("" for content that appears before the first heading)
    text            section body (heading line excluded), HTML entities decoded
    near_dup_group  id of the cross-version near-duplicate group, <NA> if none
    For_training    0 for near-duplicates outside the latest version, else 1
    is_TO_DO        1 when the body is only a placeholder ("TO DO."), else 0
"""

from __future__ import annotations

import argparse
import html
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from rag_etl.flags import DEFAULT_THRESHOLDS, flag_near_duplicates, flag_todo
from rag_etl.navigation import is_navigation_file
from rag_etl.semantic import DEFAULT_MODEL

COLUMNS = ["path", "file_name", "project_name", "version", "section", "text"]
FLAG_COLUMNS = ["near_dup_group", "For_training", "is_TO_DO"]
SECTION_SEP = " > "

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t#]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HTML_HEADING_RE = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.I | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_|`)(.+?)\1")
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


# --------------------------------------------------------------------------
# Markdown -> sections
# --------------------------------------------------------------------------


def _iter_fenced(lines: list[str]) -> Iterator[tuple[str, bool]]:
    """Yield (line, in_code) where fence delimiters count as code."""
    fence: str | None = None
    for line in lines:
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            yield line, True
        else:
            yield line, fence is not None


def _html_heading_to_atx(match: re.Match) -> str:
    level = int(match.group(1))
    title = re.sub(r"\s+", " ", _HTML_TAG_RE.sub("", match.group(2))).strip()
    return f"\n{'#' * level} {title}\n"


def html_headings_to_atx(markdown: str) -> str:
    """Rewrite ``<hN>..</hN>`` (possibly multi-line) outside code fences as
    ATX headings on their own line, so they split sections like ``#`` does."""
    out: list[str] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            out.append(_HTML_HEADING_RE.sub(_html_heading_to_atx, "\n".join(prose)))
            prose.clear()

    for line, in_code in _iter_fenced(markdown.split("\n")):
        if in_code:
            flush_prose()
            out.append(line)
        else:
            prose.append(line)
    flush_prose()
    return "\n".join(out)


def _clean_heading(raw: str) -> str:
    title = _HTML_COMMENT_RE.sub("", raw)
    title = _LINK_RE.sub(r"\1", title)
    title = _EMPHASIS_RE.sub(r"\2", title)
    return html.unescape(title).strip()


def _clean_text(lines: list[str]) -> str:
    text = _HTML_COMMENT_RE.sub("", "\n".join(lines))
    text = html.unescape(text)
    # Collapse runs of blank lines left behind by removed comments/headings.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split Markdown into (section_breadcrumb, text) pairs.

    Headings inside fenced code blocks (e.g. ``# comment`` in a bash snippet)
    are ignored. Sections whose body is empty are dropped.
    """
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    markdown = html_headings_to_atx(markdown)
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []  # (level, title) of open headings
    buffer: list[str] = []

    def flush() -> None:
        text = _clean_text(buffer)
        if text:
            sections.append((SECTION_SEP.join(t for _, t in stack), text))
        buffer.clear()

    for line, in_code in _iter_fenced(markdown.split("\n")):
        heading = None if in_code else _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, _clean_heading(heading.group(2))))
        else:
            buffer.append(line)

    flush()
    return sections


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------


def load_files(root: str | Path) -> pd.DataFrame:
    """One row per ``*.md`` file: path, file_name, project_name, version, markdown."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    rows = []
    for file_path in sorted(root.rglob("*.md")):
        rel = file_path.relative_to(root)
        if len(rel.parts) < 3:
            raise ValueError(
                f"{rel} is not under <project_name>/<version>/ relative to {root}"
            )
        rows.append(
            {
                "path": rel.as_posix(),
                "file_name": rel.name,
                "project_name": rel.parts[0],
                "version": rel.parts[1],
                "markdown": file_path.read_text(encoding="utf-8", errors="replace"),
            }
        )
    return pd.DataFrame(
        rows, columns=["path", "file_name", "project_name", "version", "markdown"]
    )


def drop_navigation_files(files: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Remove sidebars, coverpages and other link-menu files.
    ``kwargs`` are forwarded to :func:`rag_etl.navigation.is_navigation_file`."""
    is_nav = files["markdown"].map(lambda md: is_navigation_file(md, **kwargs))
    return files[~is_nav].reset_index(drop=True)


def explode_sections(files: pd.DataFrame) -> pd.DataFrame:
    """One row per section, with the base string schema (``COLUMNS``)."""
    rows = [
        {
            "path": f.path,
            "file_name": f.file_name,
            "project_name": f.project_name,
            "version": f.version,
            "section": section,
            "text": text,
        }
        for f in files.itertuples(index=False)
        for section, text in split_sections(f.markdown)
    ]
    return pd.DataFrame(rows, columns=COLUMNS).astype("string")


def build_dataset(
    root: str | Path,
    method: str = "minhash",
    dup_threshold: float | None = None,
    model=None,
) -> pd.DataFrame:
    """Run the full pipeline end to end.

    ``method`` picks the near-duplicate detector (``"minhash"`` or
    ``"semantic"``); ``dup_threshold`` defaults to that method's calibrated
    threshold; ``model`` is a SentenceTransformer (semantic only)."""
    files = drop_navigation_files(load_files(root))
    sections = explode_sections(files)
    sections = flag_todo(sections)
    kwargs = {"model": model} if method == "semantic" else {}
    sections = flag_near_duplicates(
        sections, method=method, threshold=dup_threshold, **kwargs
    )
    return sections[COLUMNS + FLAG_COLUMNS]


def save_dataset(df: pd.DataFrame, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    elif out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        raise ValueError(f"unsupported output extension: {out.suffix}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", help="Root folder: <project>/<version>/**/*.md")
    parser.add_argument(
        "-o", "--output", required=True, help="Output file (.parquet or .csv)"
    )
    parser.add_argument(
        "--method",
        choices=sorted(DEFAULT_THRESHOLDS),
        default="minhash",
        help="Near-duplicate detector: MinHash LSH or embedding cosine similarity",
    )
    parser.add_argument(
        "--dup-threshold",
        type=float,
        default=None,
        help="Similarity at/above which sections are near-duplicates "
        f"(default per method: {DEFAULT_THRESHOLDS})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model for --method semantic",
    )
    args = parser.parse_args(argv)

    model = None
    if args.method == "semantic":
        from rag_etl.semantic import load_model

        model = load_model(args.model)
    df = build_dataset(
        args.input_dir, method=args.method, dup_threshold=args.dup_threshold, model=model
    )
    out = save_dataset(df, args.output)
    print(
        f"Wrote {len(df)} sections from {df['path'].nunique()} files "
        f"({df['project_name'].nunique()} projects) to {out}: "
        f"For_training=1: {int(df['For_training'].sum())}, "
        f"is_TO_DO=1: {int(df['is_TO_DO'].sum())}"
    )


if __name__ == "__main__":
    main()
