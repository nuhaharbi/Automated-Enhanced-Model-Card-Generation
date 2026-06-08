"""Core orchestrator — entry points for all preprocessing stages.

Usage::

    from preprocessor import preprocess_paper, preprocess_markdown

    # Paper (web app — plain text, no table summarisation)
    paper = preprocess_paper(paper_text)

    # Paper (batch — marker JSON + table summarisation)
    from preprocessor.table_summarizer import TableSummarizer
    summarizer = TableSummarizer()
    paper = preprocess_paper(paper_text, paper_json=json_str,
                             summarize_tables=True, table_summarizer=summarizer)

    # Model card or GitHub README
    md = preprocess_markdown(readme_text, summarize_tables=True,
                             table_summarizer=summarizer)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from preprocessor.models import PreprocessedMarkdown, PreprocessedPaper
from preprocessor.paper_preprocessor import (
    concatenate_span_paragraphs,
    filter_unwanted_sections,
    is_english_text,
    parse_marker_json,
    parse_plain_text,
    split_into_levels,
)

if TYPE_CHECKING:
    from preprocessor.table_summarizer import TableSummarizer


logger = logging.getLogger(__name__)

_ENGLISH_CODES = {"en", "eng"}
_ENGLISH_NAMES = {"english"}


def _metadata_language_is_english(language: Any) -> bool | None:
    """Return English decision from metadata language, or ``None`` if unknown.

    Metadata-first gate:
    - ``True``  -> declared English
    - ``False`` -> declared non-English
    - ``None``  -> no reliable metadata, caller should fall back to heuristic
    """
    if language is None:
        return None

    values: list[str] = []

    if isinstance(language, str):
        values = [language]
    elif isinstance(language, (list, tuple, set)):
        values = [str(v) for v in language]
    else:
        return None

    cleaned: list[str] = []
    for v in values:
        token = v.strip().lower().replace("_", "-")
        if token:
            cleaned.append(token)

    if not cleaned:
        return None

    def _is_english_token(token: str) -> bool:
        return token in _ENGLISH_CODES or token in _ENGLISH_NAMES or token.startswith("en-")

    has_english = any(_is_english_token(t) for t in cleaned)
    if has_english:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Paper preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_paper(
    paper_content: str,
    *,
    paper_json: str = "",
    declared_language: Any = None,
    summarize_tables: bool = False,
    table_summarizer: "TableSummarizer | None" = None,
) -> PreprocessedPaper:
    """Preprocess research-paper content into structured splits.

    Pipeline:
        1. Parse content (marker JSON → structured; plain text → heuristic).
        2. Merge split-paragraph spans.
        3. Remove boilerplate sections (intro, abstract, references, …).
        4. Split into section / paragraph / sentence levels.
        5. (Optional) Summarise tables via LLM.

    Parameters
    ----------
    paper_content : str
        Paper text — either plain text (pymupdf) or, when *paper_json* is
        empty, we try to detect and parse JSON from this field as well.
    paper_json : str
        Raw marker JSON string.  When non-empty, the structured parser is
        used.  When empty, the function falls back to plain-text parsing.
    declared_language : Any
        Optional upstream language metadata (e.g. ``card_data.language``).
        If present, this metadata is used as the first-stage language gate;
        otherwise we fall back to text heuristic detection.
    summarize_tables : bool
        Whether to generate LLM descriptions for tables.  Defaults to
        ``False`` (web-app mode).  Set ``True`` for batch experiments.
    table_summarizer : TableSummarizer | None
        Pre-initialised summariser instance (reuse across calls to avoid
        reloading the model).  If *summarize_tables* is ``True`` and this
        is ``None``, a default :class:`TableSummarizer` is created.

    Returns
    -------
    PreprocessedPaper
        Structured output with sections, three split levels, and optional
        table descriptions / math / link maps.
    """
    if not paper_content and not paper_json:
        return PreprocessedPaper()

    # --- Step 0: Language check (metadata-first, heuristic fallback) --------
    lang_from_metadata = _metadata_language_is_english(declared_language)
    if lang_from_metadata is False:
        return PreprocessedPaper()

    check_text = paper_content or paper_json
    if lang_from_metadata is None and not is_english_text(check_text):
        return PreprocessedPaper()

    # --- Step 1: Decide parsing strategy ------------------------------------
    use_json = False
    json_source = paper_json

    if json_source:
        use_json = True
    else:
        # paper_content itself might be marker JSON
        try:
            probe = json.loads(paper_content)
            if isinstance(probe, dict) and "children" in probe:
                json_source = paper_content
                use_json = True
        except (json.JSONDecodeError, TypeError):
            pass

    # --- Step 2: Parse content ----------------------------------------------
    if use_json:
        sections, raw_tables, math_map, links_map = parse_marker_json(json_source)
    else:
        sections, raw_tables, math_map, links_map = parse_plain_text(paper_content)

    # --- Step 3: Post-processing --------------------------------------------
    sections = concatenate_span_paragraphs(sections)
    sections = filter_unwanted_sections(sections)

    # --- Step 4: Split into levels ------------------------------------------
    sections_split, paragraphs_split, sentences_split = split_into_levels(sections)

    # --- Step 5: (Optional) Table summarisation -----------------------------
    table_descriptions: list[str] = []
    if summarize_tables and raw_tables:
        if table_summarizer is None:
            from preprocessor.table_summarizer import TableSummarizer as _TS

            table_summarizer = _TS()
        table_descriptions = table_summarizer.summarize(raw_tables)

    # --- Step 6: Assemble result --------------------------------------------
    return PreprocessedPaper(
        sections=sections,
        sections_split=sections_split,
        paragraphs_split=paragraphs_split,
        sentences_split=sentences_split,
        table_descriptions=table_descriptions,
        math_map=math_map,
        links_map=links_map,
        success=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Markdown preprocessing (model card / GitHub README)
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_markdown(
    md_text: str,
    *,
    declared_language: Any = None,
    summarize_tables: bool = False,
    table_summarizer: "TableSummarizer | None" = None,
) -> PreprocessedMarkdown:
    """Preprocess a markdown document (model card or GitHub README).

    Pipeline:
        1. Strip emojis and shortcodes.
        2. Parse markdown AST via mistune.
        3. Walk AST → sections, paragraphs, code blocks, table placeholders.
        4. Sentence-tokenise and clean sections.
        5. (Optional) Replace ``[N:TABLE]`` placeholders with LLM summaries
           *after* tokenisation — summaries are not original text.

    Parameters
    ----------
    md_text : str
        Raw markdown content.
    declared_language : Any
        Optional upstream language metadata (e.g. ``card_data.language``).
        If present, this metadata is used as the first-stage language gate;
        otherwise we fall back to text heuristic detection.
    summarize_tables : bool
        Whether to generate LLM descriptions for markdown table nodes.
    table_summarizer : TableSummarizer | None
        Pre-initialised summariser (reuse across calls).

    Returns
    -------
    PreprocessedMarkdown
        Structured output with three split levels plus codes, links, emails.
    """
    if not md_text:
        return PreprocessedMarkdown()

    # --- Language check (metadata-first, heuristic fallback) ----------------
    lang_from_metadata = _metadata_language_is_english(declared_language)
    if lang_from_metadata is False:
        return PreprocessedMarkdown()
    if lang_from_metadata is None and not is_english_text(md_text):
        return PreprocessedMarkdown()

    from preprocessor.markdown_preprocessor import parse_markdown

    try:
        sections, paragraphs, sentences, codes, tables, links, emails = parse_markdown(
            md_text,
            summarize_tables=summarize_tables,
            table_summarizer=table_summarizer,
        )
    except Exception:
        logger.exception("Markdown preprocessing failed")
        return PreprocessedMarkdown()

    return PreprocessedMarkdown(
        sections_split=sections,
        paragraphs_split=paragraphs,
        sentences_split=sentences,
        codes=codes,
        tables=tables,
        links=links,
        emails=emails,
        success=True,
    )
