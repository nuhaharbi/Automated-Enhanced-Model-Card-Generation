"""Paper content preprocessing — parse marker JSON or plain text.

This module contains the core parsing logic for research-paper content.

* :func:`parse_marker_json` — full structured extraction from marker's
  JSON output (sections, tables, math, links).
* :func:`parse_plain_text`  — lightweight extraction from plain text
  produced by the pymupdf backend.

Both return the same tuple of ``(sections, tables, math_map, links_map)``
so that downstream functions work identically regardless of the source.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Try to import NLTK sentence tokeniser; fall back to a simple splitter.
# ---------------------------------------------------------------------------
try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

    def sent_tokenize(text: str) -> list[str]:
        try:
            return _nltk_sent_tokenize(text)
        except LookupError:
            # Auto-download punkt tokeniser data on first use
            import nltk

            nltk.download("punkt_tab", quiet=True)
            return _nltk_sent_tokenize(text)

except ImportError:

    def sent_tokenize(text: str) -> list[str]:  # type: ignore[misc]
        """Naïve fallback: split on sentence-ending punctuation."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Language detection — lightweight English heuristic
# ---------------------------------------------------------------------------
# We only consider English files. A simple ASCII-letter-ratio check is fast and
# sufficient: English text is overwhelmingly ASCII, while Chinese, Japanese,
# Korean, Arabic, etc. are not.  The threshold is intentionally lenient so
# that code-heavy READMEs and papers with occasional math/Unicode still pass.

def is_english_text(text: str, *, min_length: int = 50, ascii_ratio: float = 0.70) -> bool:
    """Return ``True`` if *text* looks like English content.

    The check is deliberately simple — count the fraction of characters
    that are basic ASCII letters (a-z, A-Z).  English prose, even mixed
    with code or markdown, typically has ≥70 % ASCII letters among its
    alphabetic characters.  CJK/Arabic/Cyrillic-dominant text falls well
    below this.

    Parameters
    ----------
    text : str
        Raw input text (model card, README, or paper).
    min_length : int
        Texts shorter than this are assumed English (too little signal).
    ascii_ratio : float
        Minimum ratio of ASCII letters to *all* Unicode letters.
    """
    if len(text) < min_length:
        return True  # too short to judge — assume English

    # Sample the first ~5 000 chars for speed (language doesn't change midway)
    sample = text[:5000]

    ascii_letters = sum(1 for c in sample if c.isascii() and c.isalpha())
    all_letters = sum(1 for c in sample if c.isalpha())

    if all_letters == 0:
        return True  # purely numeric / symbolic — let it through

    return (ascii_letters / all_letters) >= ascii_ratio


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Marker JSON parsing
# ═══════════════════════════════════════════════════════════════════════════

# Shared counters & accumulators are kept in a _ParseState to avoid globals.
class _ParseState:
    """Mutable accumulator passed through the recursive parser."""

    def __init__(self) -> None:
        self.math_map: dict[str, str] = {}
        self.links_map: dict[str, str] = {}
        self.all_sections: dict[str, list[str]] = {}
        self.table_descriptions: dict[str, list[dict[str, Any]]] = {}
        self.list_items: list[str] = []
        self.parent_section: str = ""
        self.current_heading: str = "N/A"
        self.table_counter: int = 0
        self.math_counter: int = 0
        self.link_counter: int = 0


# ---------------------------------------------------------------------------
# Helper: HTML → plain text
# ---------------------------------------------------------------------------
def _remove_html(html: str) -> str:
    """Strip HTML tags and return plain text using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")


# ---------------------------------------------------------------------------
# Helper: math / link placeholder replacement
# ---------------------------------------------------------------------------
def _replace_math(html: str, state: _ParseState) -> str:
    """Replace ``<math>`` tags with numbered placeholders."""

    def _replacer(match: re.Match) -> str:
        placeholder = f"[{state.math_counter}:MATH]"
        state.math_map[placeholder] = match.group(0)
        state.math_counter += 1
        return placeholder

    return re.sub(r"<math[^>]*>.*?</math>", _replacer, html)


_URL_RE = re.compile(r"https?://[^\s\"<>]+[^\s\"<>.,!?)]")


def _replace_links(html: str, state: _ParseState) -> str:
    """Replace bare URLs with numbered placeholders."""

    def _replacer(match: re.Match) -> str:
        placeholder = f"[{state.link_counter}:LINK]"
        state.links_map[placeholder] = match.group(0)
        state.link_counter += 1
        return placeholder

    return _URL_RE.sub(_replacer, html)


# ---------------------------------------------------------------------------
# Helper: extract raw text from a block (handles math/link replacement)
# ---------------------------------------------------------------------------
def _extract_raw_text(block: dict[str, Any], state: _ParseState) -> str:
    """Clean a single block's HTML into plain text with placeholders."""
    html: str = block.get("html", "")

    # Remove superscript footnotes
    html = re.sub(r"<sup>\d+</sup>", "", html)

    # Replace math tags
    if "</math>" in html:
        html = _replace_math(html, state)
    # Replace bare URLs 
    if _URL_RE.search(html):
        html = _replace_links(html, state)

    return _remove_html(html)


# ---------------------------------------------------------------------------
# Recursive block extractor (core of parse_marker_json)
# ---------------------------------------------------------------------------
def _extract_block(block: dict[str, Any] | None, state: _ParseState) -> None:
    """Recursively walk a marker JSON block tree, populating *state*."""
    if block is None:
        return

    block_type = block.get("block_type", "")

    # ── Section headers ────────────────────────────────────────────────
    if block_type == "SectionHeader":
        text = _remove_html(block.get("html", "")).strip().lower()
        match = re.match(r"(\d+\.?\d*)\s*(.*)", text)

        if match:
            number = match.group(1)
            name = match.group(2)

            if "." in number:
                # Subsection
                parent_num = number.split(".")[0]
                if state.parent_section == parent_num:
                    state.current_heading = (
                        state.current_heading.split("@")[0].strip() + " @ " + name
                    )
                else:
                    state.parent_section = parent_num
                    state.current_heading = name
            else:
                state.parent_section = number
                state.current_heading = name
        else:
            state.current_heading = text

        state.all_sections.setdefault(state.current_heading, [])

    # ── Tables ─────────────────────────────────────────────────────────
    elif block_type == "TableGroup":
        state.table_descriptions[f"Table{state.table_counter}"] = block.get("children", [])
        state.table_counter += 1

    # ── Text blocks ────────────────────────────────────────────────────
    elif block_type == "Text":
        text = _extract_raw_text(block, state)
        state.all_sections.setdefault(state.current_heading, [])
        state.all_sections[state.current_heading].append(text)

    # ── List groups ────────────────────────────────────────────────────
    elif block_type == "ListGroup":
        items: list[str] = []
        for item in block.get("children", []):
            items.append(_extract_raw_text(item, state))
        text = " ".join(items)
        state.all_sections.setdefault(state.current_heading, [])
        state.all_sections[state.current_heading].append(text)

    # ── Equations ──────────────────────────────────────────────────────
    elif block_type == "Equation":
        html = block.get("html", "")
        html = _replace_math(html, state)
        text = _remove_html(html)
        state.all_sections.setdefault(state.current_heading, [])
        state.all_sections[state.current_heading].append(text)

    # ── Recurse into children (except TableGroup, already handled) ─────
    if block.get("children") and block_type != "TableGroup":
        for child in block["children"]:
            _extract_block(child, state)


def parse_marker_json(
    json_data: str,
) -> tuple[dict[str, list[str]], dict[str, list[dict]], dict[str, str], dict[str, str]]:
    """Parse marker's structured JSON output into sections, tables, etc.

    Parameters
    ----------
    json_data : str
        Raw JSON string produced by the marker PDF converter.

    Returns
    -------
    sections : dict[str, list[str]]
        Mapping of ``section_heading → [text_block, …]``.
    tables : dict[str, list[dict]]
        Raw table block children, keyed ``"Table0"``, ``"Table1"``, …
    math_map : dict[str, str]
        ``"[0:MATH]"`` → original ``<math>`` tag.
    links_map : dict[str, str]
        ``"[0:LINK]"`` → original URL.
    """
    state = _ParseState()
    data = json.loads(json_data)

    for page in data.get("children", []):
        _extract_block(page, state)

    return state.all_sections, state.table_descriptions, state.math_map, state.links_map


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Plain-text fallback (for pymupdf backend)
# ═══════════════════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+\.?\d*)\s+\S"  # "1 Introduction", "2.1 Method"
    r"|[A-Z][A-Z ]{3,}$"  # "ABSTRACT", "RELATED WORK"
    r")",
    re.MULTILINE,
)


def parse_plain_text(
    text: str,
) -> tuple[dict[str, list[str]], dict[str, list[dict]], dict[str, str], dict[str, str]]:
    """Lightweight extraction from plain text (pymupdf output).

    Attempts to split the text into sections using simple heading
    heuristics.  No table or math extraction is possible from plain text,
    so those maps are returned empty.

    Returns the same 4-tuple as :func:`parse_marker_json` for API
    compatibility.
    """
    sections: dict[str, list[str]] = {}
    current_heading = "N/A"
    sections[current_heading] = []

    links_map: dict[str, str] = {}
    link_counter = 0

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Detect headings
        if _HEADING_RE.match(stripped) and len(stripped.split()) <= 10:
            heading_match = re.match(r"(\d+\.?\d*)\s+(.*)", stripped)
            if heading_match:
                current_heading = heading_match.group(2).strip().lower()
            else:
                current_heading = stripped.lower()
            sections.setdefault(current_heading, [])
        else:
            # Replace bare URLs with placeholders
            def _link_replacer(match: re.Match) -> str:
                nonlocal link_counter
                placeholder = f"[{link_counter}:LINK]"
                links_map[placeholder] = match.group(0)
                link_counter += 1
                return placeholder

            cleaned = _URL_RE.sub(_link_replacer, stripped)
            sections.setdefault(current_heading, [])
            sections[current_heading].append(cleaned)

    # No tables or math from plain text
    return sections, {}, {}, links_map


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — Post-processing (shared by both paths)
# ═══════════════════════════════════════════════════════════════════════════

# Sections to drop — introductory, boilerplate, or non-technical content.
_SECTIONS_TO_AVOID = frozenset(
    [
        "introduction",
        "abstract",
        "background",
        "related work",
        "literature review",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgments",
        "acknowledgment",
        "ablation study",
        "ablation studies",
        "conflicts of interest",
    ]
)


# Regex for detecting a proper sentence-ending character (or placeholder).
_SENTENCE_END_RE = re.compile(
    r"(?:"
    r"[.!?]"                         # standard punctuation
    r"|\]"                            # placeholder bracket — [N:MATH], [N:CODE]
    r"|[.!?][\"'\u201d\u2019)\]]*"  # punctuation followed by quotes/brackets
    r")$"
)


def concatenate_span_paragraphs(
    sections: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Merge text blocks that were split mid-sentence.

    Consecutive blocks where the first does not end with sentence-ending
    punctuation (``.``, ``!``, ``?``) or a placeholder bracket (``]``)
    are concatenated until a sentence boundary is found.  Also strips
    LaTeX escape sequences.
    """
    for _name, content in sections.items():
        idx = 0
        while idx < len(content):
            # Strip LaTeX escape sequences
            content[idx] = re.sub(r"\\[a-zA-Z0-9]+", "", content[idx])

            # Merge with next block until we hit a sentence-ending char
            while not _SENTENCE_END_RE.search(content[idx].rstrip()):
                if idx + 1 < len(content):
                    content[idx] += " " + content[idx + 1]
                    content.pop(idx + 1)
                else:
                    break
            idx += 1

    return sections


def filter_unwanted_sections(
    sections: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Remove boilerplate / non-technical sections.

    Everything before (and including) *introduction* is dropped so that
    the output begins with the paper's core contribution sections.
    """
    filtered: dict[str, list[str]] = {}

    for section_name, section_content in sections.items():
        if any(s in section_name for s in _SECTIONS_TO_AVOID):
            if section_name == "introduction":
                # Drop everything collected so far (intro + preceding)
                filtered = {}
            continue
        filtered[section_name] = section_content

    return filtered


# ---------------------------------------------------------------------------
# Abbreviation protection — shared by paper & markdown sentence splitting
# ---------------------------------------------------------------------------
_ABBREV_PAIRS: list[tuple[str, str]] = [
    ("et al.", "et al\x00"),
    ("Fig.", "Fig\x00"),
    ("Tab.", "Tab\x00"),
    ("e.g.", "e\x00g\x00"),
    ("i.e.", "i\x00e\x00"),
    ("etc.", "etc\x00"),
    ("Eq.", "Eq\x00"),
    ("Sec.", "Sec\x00"),
    ("Ref.", "Ref\x00"),
    ("No.", "No\x00"),
    ("vs.", "vs\x00"),
    ("Dr.", "Dr\x00"),
    ("Prof.", "Prof\x00"),
    ("approx.", "approx\x00"),
    ("incl.", "incl\x00"),
    ("Vol.", "Vol\x00"),
]


def protect_abbreviations(text: str) -> str:
    """Replace common abbreviation periods with NUL so they survive sentence splitting."""
    for orig, safe in _ABBREV_PAIRS:
        text = text.replace(orig, safe)
    return text


def restore_abbreviations(text: str) -> str:
    """Reverse :func:`protect_abbreviations` — restore original periods."""
    return text.replace("\x00", ".")


# ---------------------------------------------------------------------------
# Pipe-delimited table detection — smarter than bare ``"|" in text``
# ---------------------------------------------------------------------------
_PIPE_TABLE_RE = re.compile(
    r"(?:^\|.*\|$)"         # starts and ends with |
    r"|(?:.*\|.*\|.*\|)",   # or contains 3+ pipe characters
    re.MULTILINE,
)


def looks_like_table_row(text: str) -> bool:
    """Return True if *text* looks like a pipe-delimited table row.

    A string is considered table-like when it *starts and ends* with ``|``
    OR contains **three or more** pipe characters — both strong indicators
    of markdown table syntax.  Lone uses of ``|`` ("input | output") are
    not flagged.
    """
    stripped = text.strip()
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    return stripped.count("|") >= 3

# ---------------------------------------------------------------------------
# Citation patterns — used by clean_text_list()
# ---------------------------------------------------------------------------
_CITE_BRACKET_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")
_CITE_INLINE_RE = re.compile(
    r"\("
    r"(?:[A-Za-z][A-Za-z .,&-]*?\s+et\s+al\.?"
    r"|[A-Za-z][A-Za-z .,&-]*?)?"
    r",?\s*\d{4}[a-z]?"
    r"(?:;\s*[A-Za-z][A-Za-z .,&-]*?,?\s*\d{4}[a-z]?)*"
    r"\)"
)


def clean_text_list(items: list[str]) -> list[str]:
    """Post-clean text items: remove citations, strip quotes, normalise whitespace.

    Applied as a final pass to section / paragraph / sentence lists from
    both the paper and markdown preprocessors.

    Steps
    -----
    1. Remove bracketed citation numbers — ``[1]``, ``[1, 2, 3]``.
    2. Remove inline author-year citations — ``(Smith et al., 2020)``,
       ``(Jones, 2019; Smith et al., 2020a)``.
    3. Strip wrapping single quotes (JSON array artifact).
    4. Collapse consecutive whitespace and strip.

    Returns only non-empty strings.
    """
    result: list[str] = []
    for s in items:
        s = _CITE_BRACKET_RE.sub("", s)
        s = _CITE_INLINE_RE.sub("", s)
        s = re.sub(r"^\s*'(.*?)',?\s*$", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Bibliography/reference-like line filtering
# ---------------------------------------------------------------------------
_BIB_PATTERNS = [
    r"\(\d{4}\)",                    # (2020)
    r"\d{4}\.",                      # 2020.
    r"\bet al\b",                    # et al.
    r"\bdoi\b",                      # doi
    r"\barxiv\b",                    # arxiv
    r"\bpp\.\b",                    # pp.
    r"\bvol\.\b",                   # vol.
    r"In Proceedings of",              # conferences
    r"Conference on",                  # conference names
    r"Journal of",                     # journal titles
    r"http[s]?://",                    # URLs
    r"\b[A-Z][a-z]+\s+[A-Z]\.\b",  # Author format e.g. "Smith J."
    r"[\w-]+,\s*\d{4}",             # e.g., "Smith, 2021"
]

_BIB_RE = re.compile("|".join(_BIB_PATTERNS), flags=re.IGNORECASE)


def remove_bibliography_strings(items: list[str]) -> list[str]:
    """Remove strings that look like bibliography/reference entries.

    A string is removed when it matches bibliography-like patterns and is
    additionally long or comma-dense (heuristic signal for reference lines).
    """
    cleaned: list[str] = []
    for s in items:
        if _BIB_RE.search(s):
            if s.count(",") >= 3 or len(s) > 100:
                continue
        cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# HuggingFace YAML metadata detection — used by remove_metadata_anywhere()
# ---------------------------------------------------------------------------
_META_KEYS = [
    "tags:", "base_model:", "language:", "model-index:",
    "dataset:", "metrics:", "task:", "revision:", "split:",
    "config:", "results:", "type:", "value:", "name:",
    "license:", "library_name:", "pipeline_tag:", "datasets:", "thumbnail:",
    "widget:", "inference:", "co2_eq_emissions:", "model_name:",
]

_META_ANYWHERE_RE = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in _META_KEYS) + r")",
    flags=re.IGNORECASE,
)


def remove_metadata_anywhere(
    items: list[str],
    *,
    min_keyword_hits: int = 2,
) -> list[str]:
    """Drop items that look like leaked HuggingFace YAML front-matter.

    Many HF model cards start with a YAML block (``---\nlanguage: en\n---``).
    After markdown parsing the YAML keys often leak into the first section
    as plain text (e.g. ``"language: en tags: exbert license: apache-2.0"``),
    which is not natural language and would confuse downstream classifiers.

    An item is removed when it contains *min_keyword_hits* or more metadata
    keywords from the standard HF YAML schema.

    Parameters
    ----------
    items : list[str]
        Section / paragraph / sentence strings.
    min_keyword_hits : int
        Minimum number of distinct keyword matches to trigger removal
        (default ``2``).
    """
    result: list[str] = []
    for s in items:
        hits = len(_META_ANYWHERE_RE.findall(s))
        if hits >= min_keyword_hits:
            continue
        result.append(s)
    return result


def split_into_levels(
    sections: dict[str, list[str]],
) -> tuple[list[str], list[str], list[str]]:
    """Split cleaned sections into three granularity levels.

    Returns
    -------
    sections_split : list[str]
        One string per section (all paragraphs joined).
    paragraphs_split : list[str]
        One string per paragraph (≥4 words, no tables).
    sentences_split : list[str]
        One string per sentence (≥4 words, no placeholders).
    """
    sections_split: list[str] = []
    paragraphs_split: list[str] = []
    sentences_split: list[str] = []

    # Regexes for placeholder-only content
    _placeholder_only = re.compile(r"^\[\d+:(CODE|MATH|LINK)\]$")

    for _name, content in sections.items():
        if not content:
            continue

        # Section level — join non-table paragraphs
        section_texts = [t for t in content if not looks_like_table_row(t)]
        if section_texts:
            sections_split.append(" ".join(section_texts))

        for item in content:
            # Paragraph level
            if len(item.split()) > 3 and not looks_like_table_row(item):
                paragraphs_split.append(item)

            # Sentence level — protect common abbreviations
            safe = protect_abbreviations(item)
            for sent in sent_tokenize(safe):
                sent = restore_abbreviations(sent).strip()
                if (
                    not looks_like_table_row(sent)
                    and not _placeholder_only.match(sent)
                    and len(sent.split()) > 3
                ):
                    sentences_split.append(sent)

    # Post-clean: remove citations, normalise whitespace
    sections_split = clean_text_list(sections_split)
    paragraphs_split = clean_text_list(paragraphs_split)
    sentences_split = clean_text_list(sentences_split)

    # Remove bibliography/reference-like entries
    sections_split = remove_bibliography_strings(sections_split)
    paragraphs_split = remove_bibliography_strings(paragraphs_split)
    sentences_split = remove_bibliography_strings(sentences_split)

    return sections_split, paragraphs_split, sentences_split
