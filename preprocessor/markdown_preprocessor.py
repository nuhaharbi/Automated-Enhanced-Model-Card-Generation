"""Markdown file preprocessing — parse model cards and GitHub READMEs.

This module provides :func:`parse_markdown`, the core function that turns
a raw markdown string (HuggingFace model card or GitHub README) into
structured section / paragraph / sentence splits with extracted code
blocks, links, and emails.

The implementation mirrors the original ``preprocess_markdown`` logic:

1. Strip emoji characters and shortcodes.
2. Parse the markdown into a mistune AST.
3. Walk the AST to extract text, tables, and code blocks; replace
   URLs and emails with ``[N:LINK]`` / ``[N:EMAIL]`` placeholders.
   Tables become ``[N:TABLE]`` placeholders; raw nodes are collected.
4. Flush text into section-level and paragraph-level buffers on heading
   boundaries.
5. Sentence-tokenise the joined sections.
6. Post-clean — remove citations, normalise whitespace.
7. Drop leaked YAML metadata items (``tags:``, ``license:``, etc.).
8. Remove bibliography/reference-like strings.
9. (Optional) Replace ``[N:TABLE]`` placeholders in sections and
    paragraphs with LLM-generated summaries, and append each table
    summary to the sentence list as a single atomic item.  This happens
    *after* all cleaning so generated descriptions are never sentence-
    split or citation-cleaned — they are not original text.

Table nodes can optionally be summarised by a :class:`TableSummarizer`
instance (passed in from the caller).  In markdown (unlike papers) we
know the exact inline position of each table, so the summary is
inserted at the placeholder location.  Paper tables float (LaTeX) and
are kept in a separate ``table_descriptions`` list instead.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from preprocessor.table_summarizer import TableSummarizer

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so unit tests don't need them.
# ---------------------------------------------------------------------------
try:
    import regex as _regex
except ImportError:
    _regex = None  # type: ignore[assignment]

try:
    import emoji as _emoji
except ImportError:
    _emoji = None  # type: ignore[assignment]

try:
    import mistune
    # mistune 2.x uses plugin_table, older versions use table
    try:
        from mistune.plugins.table import plugin_table as _mistune_table_plugin
    except ImportError:
        from mistune.plugins.table import table as _mistune_table_plugin
except ImportError:
    mistune = None  # type: ignore[assignment]
    _mistune_table_plugin = None

# Reuse the project's shared utilities
from preprocessor.paper_preprocessor import (
    clean_text_list,
    looks_like_table_row,
    protect_abbreviations,
    remove_bibliography_strings,
    remove_metadata_anywhere,
    restore_abbreviations,
    sent_tokenize,
)

# ---------------------------------------------------------------------------
# Regex patterns for URL and email detection
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://[^\s\"<>]+[^\s\"<>.,!?)]")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def parse_markdown(
    md_text: str,
    *,
    summarize_tables: bool = False,
    table_summarizer: "TableSummarizer | None" = None,
) -> tuple[list[str], list[str], list[str], list[Any], list[Any], list[str], list[str]]:
    """Parse a markdown string into structured splits.

    Parameters
    ----------
    md_text : str
        Raw markdown content (model card or README).
    summarize_tables : bool
        Whether to generate LLM descriptions for ``table`` nodes.
    table_summarizer : TableSummarizer | None
        Pre-initialised summariser (reuse across calls).

    Returns
    -------
    sections : list[str]
        One string per heading-delimited section.
    paragraphs : list[str]
        One string per paragraph.
    sentences : list[str]
        One string per sentence (≥4 words, no placeholders), plus one
        atomic sentence item per table summary when summarisation is
        enabled.
    codes : list[Any]
        Raw ``block_code`` AST nodes.
    tables : list[Any]
        Raw ``table`` AST nodes in order of appearance.
    links : list[str]
        Extracted URLs in order of appearance.
    emails : list[str]
        Extracted email addresses in order of appearance.
    """
    if not md_text:
        return [], [], [], [], [], [], []

    # --- Step 1: Strip emojis -----------------------------------------------
    md_text = _strip_emojis(md_text)

    # --- Step 1b: Strip YAML frontmatter ------------------------------------
    # Raw README.md files often start with ---\nkey: value\n---
    # This must be removed before markdown parsing or it leaks as text.
    md_text = _strip_yaml_frontmatter(md_text)

    # --- Step 2: Parse markdown AST -----------------------------------------
    if mistune is None:
        raise ImportError(
            "mistune is required for markdown preprocessing. "
            "Install it with: pip install mistune"
        )

    md_parser = mistune.create_markdown(
        renderer="ast", plugins=[_mistune_table_plugin]
    )
    ast_nodes = md_parser(md_text)

    # --- Step 3: Walk AST and build splits ----------------------------------
    # Links and emails are collected *during* AST walking (not before
    # parsing) to avoid placeholder characters interfering with mistune.
    sections: list[str] = []
    paragraphs: list[str] = []
    links_list: list[str] = []
    emails_list: list[str] = []
    codes_list: list[Any] = []
    tables_list: list[Any] = []
    code_counter = 0
    table_counter = 0
    link_counter = 0
    email_counter = 0

    sections_buffer: list[dict] = []
    paragraphs_buffer: list[dict] = []

    def _extract_text(nodes: list) -> str:
        """Recursively extract plain text from AST children.

        Link nodes are replaced with ``[N:LINK]`` placeholders and
        their URLs are appended to *links_list*.  Bare URLs and emails
        inside text nodes are similarly replaced with ``[N:LINK]`` /
        ``[N:EMAIL]``.
        """
        nonlocal link_counter, email_counter
        text = ""
        for node in nodes:
            if not isinstance(node, dict):
                continue

            if node["type"] == "text":
                # mistune 2.x uses "text" key, older versions use "raw"
                raw = node.get("text", "") or node.get("raw", "")

                # Replace bare URLs with [N:LINK] placeholders
                def _url_repl(m: re.Match) -> str:
                    nonlocal link_counter
                    links_list.append(m.group(0))
                    ph = f"[{link_counter}:LINK]"
                    link_counter += 1
                    return ph

                # Replace bare emails with [N:EMAIL] placeholders
                def _email_repl(m: re.Match) -> str:
                    nonlocal email_counter
                    emails_list.append(m.group(0))
                    ph = f"[{email_counter}:EMAIL]"
                    email_counter += 1
                    return ph

                raw = _URL_RE.sub(_url_repl, raw)
                raw = _EMAIL_RE.sub(_email_repl, raw)
                text += raw + " "

            elif node["type"] == "codespan":
                # mistune 2.x uses "text" key, older versions use "raw"
                text += (node.get("text", "") or node.get("raw", "")) + " "

            elif node["type"] == "link":
                url = node.get("attrs", {}).get("url", "")
                if url and url.startswith("mailto:"):
                    email_addr = url[7:]
                    if _EMAIL_RE.match(email_addr):
                        emails_list.append(email_addr)
                        text += f"[{email_counter}:EMAIL] "
                        email_counter += 1
                        continue
                if url and _URL_RE.match(url):
                    links_list.append(url)
                    text += f"[{link_counter}:LINK] "
                    link_counter += 1
                elif "children" in node:
                    text += _extract_text(node["children"])

            elif node["type"] == "image":
                continue

            elif "children" in node:
                text += _extract_text(node["children"])
        return text

    def _clean_markdown(nodes: list) -> str:
        """Convert a list of AST nodes into a single cleaned string."""
        nonlocal code_counter, table_counter

        plain: list[str] = []

        for node in nodes:
            if not isinstance(node, dict):
                continue

            if node["type"] == "table":
                # Insert a positional placeholder — the raw node is
                # collected so it can be summarised *after* all
                # tokenisation / cleaning (summaries are LLM-generated
                # text and must not be sentence-split or citation-cleaned).
                plain.append(f"[{table_counter}:TABLE].")
                tables_list.append(node)
                table_counter += 1

            elif node["type"] == "block_code":
                plain.append(f"[{code_counter}:CODE].")
                codes_list.append(node)
                code_counter += 1

            elif "children" in node:
                plain.append(_extract_text(node["children"]))

        text = "\n".join(plain).replace("\r\n", "\n").replace("..", ".")

        # Normalise Unicode (NFKD decomposes ligatures/special forms),
        # then remove only control characters — NOT accented letters,
        # CJK, Arabic, etc.
        import unicodedata

        text = unicodedata.normalize("NFKD", text)

        for pat, repl in [
            (r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", ""),  # control chars only
            (r"(?<!\n)\n(?!\n)", " "),
            (r" {2,}", " "),
            (r"(\S)\s\.", r"\1."),
            (r"\s+", " "),
            (r"^\s*(?:[^\w\s]|_)(?:\s+(?:[^\w\s]|_))*\s*$", ""),
        ]:
            text = re.sub(pat, repl, text).strip()

        return text

    def _flush(buffer: list[dict], split_type: str) -> None:
        """Flush the accumulated buffer into the appropriate split list."""
        if not buffer:
            return
        text = _clean_markdown(buffer)
        # Let [N:TABLE] placeholders through — they'll be replaced later.
        is_table_ph = bool(re.search(r"\[\d+:TABLE\]", text))
        if not is_table_ph:
            if looks_like_table_row(text) or len(text.split()) <= 3:
                return
            if re.match(r"^\[\d+:CODE\]\.*$", text):
                return
        if split_type == "sec":
            if not text.endswith("."):
                text += "."
            sections.append(text)
        elif split_type == "par":
            paragraphs.append(text)

    # --- Walk top-level AST nodes -------------------------------------------
    for node in ast_nodes:
        if not isinstance(node, dict):
            continue

        if node["type"] == "paragraph":
            _flush(paragraphs_buffer, "par")
            paragraphs_buffer = []

        elif node["type"] == "heading":
            _flush(paragraphs_buffer, "par")
            _flush(sections_buffer, "sec")
            paragraphs_buffer = []
            sections_buffer = []

        if node["type"] != "blank_line":
            paragraphs_buffer.append(node)
            if node["type"] != "heading":
                sections_buffer.append(node)

    # Flush remaining buffers
    _flush(paragraphs_buffer, "par")
    _flush(sections_buffer, "sec")

    # --- Step 4: Sentence-tokenise ------------------------------------------
    sentences = _sentence_split(sections)

    # --- Step 5: Post-clean — remove citations, normalise whitespace --------
    sections = clean_text_list(sections)
    paragraphs = clean_text_list(paragraphs)
    sentences = clean_text_list(sentences)

    # --- Step 6: Drop leaked YAML metadata items ----------------------------
    sections = remove_metadata_anywhere(sections)
    paragraphs = remove_metadata_anywhere(paragraphs)
    sentences = remove_metadata_anywhere(sentences)

    # --- Step 7: Remove bibliography/reference-like strings -----------------
    sections = remove_bibliography_strings(sections)
    paragraphs = remove_bibliography_strings(paragraphs)
    sentences = remove_bibliography_strings(sentences)

    # --- Step 8: Replace [N:TABLE] placeholders with summaries --------------
    # Summaries are inserted *after* all tokenisation / cleaning because
    # they are LLM-generated text — not original content — and should
    # not be sentence-split or citation-cleaned.
    if summarize_tables and table_summarizer is not None and tables_list:
        summaries: dict[str, str] = {}
        for idx, tbl_node in enumerate(tables_list):
            desc = table_summarizer.summarize_single(tbl_node)
            desc = re.sub(r"\s+", " ", str(desc)).strip()
            if desc and desc[-1] not in ".!?":
                desc = desc + "."
            summaries[f"[{idx}:TABLE]"] = desc

        def _replace_table_placeholders(items: list[str]) -> list[str]:
            result: list[str] = []
            for item in items:
                for ph, desc in summaries.items():
                    # Replace "[N:TABLE]." (with trailing period from flush)
                    item = item.replace(ph + ".", desc)
                    item = item.replace(ph, desc)
                result.append(item)
            return result

        sections = _replace_table_placeholders(sections)
        paragraphs = _replace_table_placeholders(paragraphs)

        # Add table summaries to sentences as atomic items (unsplit).
        # We append them directly rather than running sentence tokenisation
        # on generated text.
        for idx in range(len(tables_list)):
            summary = summaries.get(f"[{idx}:TABLE]", "")
            if summary:
                sentences.append(summary)

    return sections, paragraphs, sentences, codes_list, tables_list, links_list, emails_list


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _strip_emojis(text: str) -> str:
    """Remove Unicode emoji characters and ``:shortcode:`` patterns."""
    if _regex is not None and _emoji is not None:
        graphemes = _regex.findall(r"\X", text)
        text = "".join(g for g in graphemes if not _emoji.is_emoji(g))
    # Also remove :shortcode: emoji patterns
    text = re.sub(r":[a-zA-Z0-9_+-]+:", "", text)
    return text


# Regex to match YAML frontmatter: starts with ---, ends with --- or ...
_YAML_FRONTMATTER_RE = re.compile(
    r"\A\s*---\s*\n.*?\n(?:---|\.\.\.)[ \t]*\n?",
    flags=re.DOTALL,
)


def _strip_yaml_frontmatter(text: str) -> str:
    """Remove YAML frontmatter block from the start of markdown text.

    HuggingFace model cards typically start with::

        ---
        language: en
        tags:
          - text-classification
        license: apache-2.0
        ---

    This block must be removed before markdown parsing or the YAML keys
    will leak into the parsed content as plain text.
    """
    return _YAML_FRONTMATTER_RE.sub("", text)


def _sentence_split(sections: list[str]) -> list[str]:
    """Sentence-tokenise joined section text, filtering short / placeholder lines."""
    all_content = " ".join(sections)

    safe = protect_abbreviations(all_content)

    sentences: list[str] = []
    for s in sent_tokenize(safe):
        s = restore_abbreviations(s).strip()
        if (
            len(s.split()) > 3
            and not re.match(r"^\[\d+:(CODE|TABLE)\]\.*$", s)
            and not looks_like_table_row(s)
        ):
            sentences.append(s)

    return sentences
