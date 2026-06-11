"""Sample-card loading and placeholder resolution helpers for the web app."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from classifier.models import ClassificationResult
from preprocessor.core import preprocess_paper
from retriever.core import retrieve_model_artifacts

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_GENERATED_CARDS_PATH = _PROJECT_ROOT / "outputs" / "full_pipeline_eval" / "task_stratified_generated_cards.jsonl"
_SAMPLE_REPRO_REPORTS_PATH = _PROJECT_ROOT / "outputs" / "full_pipeline_eval" / "task_stratified_reproducibility_reports_full.jsonl"
_PREVIEW_DATASET_DIR = _PROJECT_ROOT / "sample_preview"
_POOL_CSV_PATH = _PREVIEW_DATASET_DIR / "new_all_repos_both_links.csv"
_PREPROCESSED_MC_CSV_PATH = _PREVIEW_DATASET_DIR / "new_preprocessed_mc.csv"
_PREPROCESSED_PAPERS_CSV_PATH = _PREVIEW_DATASET_DIR / "new_preprocessed_papers_all.csv"
_PREPROCESSED_GITHUB_CSV_PATH = _PREVIEW_DATASET_DIR / "new_preprocessed_github.csv"

_PLACEHOLDER_SOURCE_CACHE: dict[str, Any] | None = None
_REPRO_SCORE_CACHE: dict[str, float] | None = None
_REPRO_REPORT_CACHE: dict[str, dict[str, Any]] | None = None
_REPRO_REPORT_CACHE_MTIME: float | None = None

_SECTION_AWARE_PLACEHOLDER_PATTERN = re.compile(r"\[(?:(GH|MC|RP)_)?(\d+):(LINK|EMAIL|CODE|MATH|TABLE)\]")
_SECTION_AWARE_DOLLAR_PLACEHOLDER_PATTERN = re.compile(
    r"\$(?:(GH|MC|RP)_)?(\d+):(LINK|EMAIL|CODE|MATH|TABLE)\$",
    re.IGNORECASE,
)


def _stringify_structured_node(value: Any, *, max_len: int = 2000) -> str:
    if isinstance(value, dict):
        node_type = str(value.get("type", "")).strip().lower()
        if node_type == "block_code":
            raw = str(value.get("raw", value.get("text", "")))
            attrs = value.get("attrs", {})
            info = ""
            if isinstance(attrs, dict):
                info = str(attrs.get("info", "")).strip()
            if not info:
                info = str(value.get("info", value.get("language", ""))).strip()
            marker = str(value.get("marker", "```") or "```")
            marker = "```" if not marker.strip() else marker.strip()
            code_body = raw.rstrip("\n")
            return f"{marker}{info}\n{code_body}\n{marker}" if code_body else f"{marker}{info}\n{marker}"

    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            text = str(value)
    return text if len(text) <= max_len else (text[: max_len - 3] + "...")


def _merge_placeholder_value(mapping: dict[str, str], key: str, value: str) -> None:
    v = value.strip()
    if not v:
        return
    if key not in mapping:
        mapping[key] = v
        return
    return


def _parse_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return []

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    if s.startswith("[") and s.endswith("]"):
        for parser in (json.loads,):
            try:
                parsed = parser(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:  # noqa: BLE001
                pass

        try:
            import ast

            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:  # noqa: BLE001
            pass

    return [s]


def _parse_json_or_python(value: Any, default: Any) -> Any:
    if value is None:
        return default

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return default

    for parser in (json.loads,):
        try:
            return parser(s)
        except Exception:  # noqa: BLE001
            pass

    try:
        import ast

        return ast.literal_eval(s)
    except Exception:  # noqa: BLE001
        return default


def _normalize_link(url: Any) -> str:
    s = str(url or "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    s = s.split("#", 1)[0].split("?", 1)[0]
    s = s.replace("http://", "https://").rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    s = s.replace("https://arxiv.org/pdf/", "https://arxiv.org/abs/").removesuffix(".pdf")
    return s.lower()


def _resolve_link_value(value: Any) -> str:
    vals = _parse_text_list(value)
    candidate = vals[0] if vals else value
    return _normalize_link(candidate)


def _resolve_raw_link_value(value: Any) -> str:
    vals = _parse_text_list(value)
    candidate = vals[0] if vals else value
    s = str(candidate or "").strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def _build_placeholder_map_for_source(prefix: str, payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}

    for i, link in enumerate(payload.get("links", []) or []):
        out[f"[{prefix}_{i}:LINK]"] = _stringify_structured_node(link)
    for i, email in enumerate(payload.get("emails", []) or []):
        out[f"[{prefix}_{i}:EMAIL]"] = _stringify_structured_node(email)
    for i, code in enumerate(payload.get("codes", []) or []):
        out[f"[{prefix}_{i}:CODE]"] = _stringify_structured_node(code)
    for i, table in enumerate(payload.get("tables", []) or []):
        out[f"[{prefix}_{i}:TABLE]"] = _stringify_structured_node(table)
    for i, math_val in enumerate(payload.get("math", []) or []):
        out[f"[{prefix}_{i}:MATH]"] = _stringify_structured_node(math_val)

    return out


_LEGACY_PLACEHOLDER_PATTERN = re.compile(r"\[(\d+):(LINK|EMAIL|CODE|MATH|TABLE)\]")
_PLACEHOLDER_PATTERN = re.compile(r"\[((?:[A-Z]+_)?\d+):(LINK|EMAIL|CODE|MATH|TABLE)\]")
_CITATION_TAG_PATTERN = re.compile(r"\[(GH|RP|HF)\]")
_LEGACY_BARE_PLACEHOLDER_PATTERN = re.compile(r"\[(\d+):(LINK|EMAIL|CODE|MATH|TABLE)\]")
_DOLLAR_PLACEHOLDER_PATTERN = re.compile(
    r"\$((?:[A-Z]+_)?\d+):(LINK|EMAIL|CODE|MATH|TABLE)\$",
    re.IGNORECASE,
)


def _namespace_placeholders_in_text(
    text: str,
    *,
    source: str,
    source_placeholder_maps: dict[str, dict[str, str]],
) -> str:
    if not text:
        return text

    token_map = source_placeholder_maps.get(source, {})
    if not token_map:
        return text

    def repl(match: re.Match[str]) -> str:
        raw = f"[{match.group(1)}:{match.group(2)}]"
        return token_map.get(raw, raw)

    return _LEGACY_PLACEHOLDER_PATTERN.sub(repl, text)


def _add_source_placeholder(
    *,
    source: str,
    prefix: str,
    raw_token: str,
    token_type: str,
    value: str,
    source_maps: dict[str, dict[str, str]],
    replacements: dict[str, str],
    seen_values: dict[str, dict[str, dict[str, str]]],
    counters: dict[str, dict[str, int]],
) -> None:
    cleaned = value.strip()
    if not cleaned:
        return

    source_maps.setdefault(source, {})
    seen_values.setdefault(source, {})
    seen_values[source].setdefault(token_type, {})
    counters.setdefault(source, {})
    counters[source].setdefault(token_type, 0)

    canonical = seen_values[source][token_type].get(cleaned)
    if canonical is None:
        canonical = f"[{prefix}_{counters[source][token_type]}:{token_type}]"
        counters[source][token_type] += 1
        seen_values[source][token_type][cleaned] = canonical
        _merge_placeholder_value(replacements, canonical, cleaned)

    source_maps[source][raw_token] = canonical


def _build_placeholder_replacements(
    *,
    paper: Any | None,
    model_card: Any | None,
    github: Any | None,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    mapping: dict[str, str] = {}
    source_maps: dict[str, dict[str, str]] = {}
    seen_values: dict[str, dict[str, dict[str, str]]] = {}
    counters: dict[str, dict[str, int]] = {}

    if paper is not None:
        for k, v in getattr(paper, "links_map", {}).items():
            _add_source_placeholder(
                source="paper",
                prefix="RP",
                raw_token=str(k),
                token_type="LINK",
                value=_stringify_structured_node(v),
                source_maps=source_maps,
                replacements=mapping,
                seen_values=seen_values,
                counters=counters,
            )
        for k, v in getattr(paper, "math_map", {}).items():
            _add_source_placeholder(
                source="paper",
                prefix="RP",
                raw_token=str(k),
                token_type="MATH",
                value=_stringify_structured_node(v),
                source_maps=source_maps,
                replacements=mapping,
                seen_values=seen_values,
                counters=counters,
            )

    def add_markdown_placeholders(md_obj: Any | None, *, source: str, prefix: str) -> None:
        if md_obj is None:
            return
        for i, link in enumerate(getattr(md_obj, "links", [])):
            _add_source_placeholder(
                source=source,
                prefix=prefix,
                raw_token=f"[{i}:LINK]",
                token_type="LINK",
                value=_stringify_structured_node(link),
                source_maps=source_maps,
                replacements=mapping,
                seen_values=seen_values,
                counters=counters,
            )
        for i, email in enumerate(getattr(md_obj, "emails", [])):
            _add_source_placeholder(
                source=source,
                prefix=prefix,
                raw_token=f"[{i}:EMAIL]",
                token_type="EMAIL",
                value=_stringify_structured_node(email),
                source_maps=source_maps,
                replacements=mapping,
                seen_values=seen_values,
                counters=counters,
            )
        for i, code in enumerate(getattr(md_obj, "codes", [])):
            _add_source_placeholder(
                source=source,
                prefix=prefix,
                raw_token=f"[{i}:CODE]",
                token_type="CODE",
                value=_stringify_structured_node(code),
                source_maps=source_maps,
                replacements=mapping,
                seen_values=seen_values,
                counters=counters,
            )
        for i, table in enumerate(getattr(md_obj, "tables", [])):
            _add_source_placeholder(
                source=source,
                prefix=prefix,
                raw_token=f"[{i}:TABLE]",
                token_type="TABLE",
                value=_stringify_structured_node(table),
                source_maps=source_maps,
                replacements=mapping,
                seen_values=seen_values,
                counters=counters,
            )

    add_markdown_placeholders(model_card, source="model_card", prefix="MC")
    add_markdown_placeholders(github, source="github_readme", prefix="GH")
    return mapping, source_maps


def _infer_source_prefix_from_context(text: str, token_start: int, token_end: int) -> str | None:
    if not text:
        return None

    source_map = {
        "GH": "GH",
        "RP": "RP",
        "HF": "MC",
    }

    line_start = text.rfind("\n", 0, token_start)
    line_start = 0 if line_start < 0 else line_start + 1
    line_end = text.find("\n", token_end)
    line_end = len(text) if line_end < 0 else line_end
    line_text = text[line_start:line_end]

    line_tags = [m.group(1) for m in _CITATION_TAG_PATTERN.finditer(line_text)]
    if line_tags:
        return source_map.get(line_tags[-1])

    window = 220
    left = max(0, token_start - window)
    right = min(len(text), token_end + window)
    win_text = text[left:right]

    best_tag: str | None = None
    best_dist: int | None = None
    for m in _CITATION_TAG_PATTERN.finditer(win_text):
        center = left + (m.start() + m.end()) // 2
        token_center = (token_start + token_end) // 2
        dist = abs(center - token_center)
        if best_dist is None or dist < best_dist:
            best_tag = m.group(1)
            best_dist = dist

    if best_tag:
        return source_map.get(best_tag)

    return None


def _upgrade_legacy_placeholders_with_citations(text: str) -> str:
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        idx = match.group(1)
        token_type = match.group(2)
        prefix = _infer_source_prefix_from_context(text, match.start(), match.end())

        if prefix is None:
            return match.group(0)

        return f"[{prefix}_{idx}:{token_type}]"

    return _LEGACY_BARE_PLACEHOLDER_PATTERN.sub(repl, text)


def _normalize_sample_placeholder_prefixes(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"(?<=\[)PP_(?=\d+:)", "RP_", text)
    text = re.sub(r"(?<=\$)PP_(?=\d+:)", "RP_", text)
    return text


def _strip_unresolved_code_placeholders(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"\s*\[(?:PP|RP)_\d+:CODE\]\s*", " ", text)
    text = re.sub(r"```[^\n`]*\n\s*```", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_code_body(code_block: str) -> str:
    stripped = code_block.strip()
    if not stripped.startswith("```"):
        return stripped

    first_newline = stripped.find("\n")
    last_fence = stripped.rfind("```")
    if first_newline < 0 or last_fence <= first_newline:
        return stripped

    return stripped[first_newline + 1:last_fence].strip("\n")


def _is_inside_fenced_code_block(text: str, token_start: int) -> bool:
    prefix = text[:token_start]
    fence_count = len(re.findall(r"(?m)^\s*```", prefix))
    return fence_count % 2 == 1


def _apply_placeholder_replacements(text: str, replacements: dict[str, str]) -> str:
    if not text or not replacements:
        return text

    def repl(match: re.Match[str]) -> str:
        token_type = match.group(2).upper()
        token = f"[{match.group(1)}:{token_type}]"
        replacement = replacements.get(token, token)

        if token_type == "CODE" and "```" in replacement:
            return f"\n\n{replacement.strip()}\n\n"

        return replacement

    replaced = _PLACEHOLDER_PATTERN.sub(repl, text)
    replaced = _DOLLAR_PLACEHOLDER_PATTERN.sub(repl, replaced)
    return replaced


def _collect_generation_references(
    cls: ClassificationResult,
    *,
    source_placeholder_maps: dict[str, dict[str, str]] | None = None,
) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}

    all_preds = (
        cls.paper.predictions
        + cls.model_card.predictions
        + cls.github_readme.predictions
    )

    for pred in all_preds:
        label = pred.predicted_label
        paragraph = pred.paragraph
        if source_placeholder_maps:
            paragraph = _namespace_placeholders_in_text(
                paragraph,
                source=pred.source,
                source_placeholder_maps=source_placeholder_maps,
            )

        refs.setdefault(label, []).append(f"{paragraph} [Source: {pred.source}]")

    return refs


def _build_keyed_row_cache(frame: Any, *, key_fn: Any, value_builder: Any) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        key = str(key_fn(row_dict)).strip()
        cache[key] = value_builder(row_dict)
    return cache


def _load_placeholder_source_cache() -> dict[str, Any]:
    global _PLACEHOLDER_SOURCE_CACHE
    if _PLACEHOLDER_SOURCE_CACHE is not None:
        return _PLACEHOLDER_SOURCE_CACHE

    import pandas as pd

    for path in (
        _POOL_CSV_PATH,
        _PREPROCESSED_MC_CSV_PATH,
        _PREPROCESSED_PAPERS_CSV_PATH,
        _PREPROCESSED_GITHUB_CSV_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required placeholder source CSV not found: {path}")

    pool = pd.read_csv(_POOL_CSV_PATH)
    mc = pd.read_csv(_PREPROCESSED_MC_CSV_PATH)
    papers = pd.read_csv(_PREPROCESSED_PAPERS_CSV_PATH)
    github = pd.read_csv(_PREPROCESSED_GITHUB_CSV_PATH)

    mc_sources = _build_keyed_row_cache(
        mc,
        key_fn=lambda d: d.get("modelId", ""),
        value_builder=lambda d: {
            "links": _parse_json_or_python(d.get("links"), []),
            "emails": _parse_json_or_python(d.get("emails"), []),
            "codes": _parse_json_or_python(d.get("codes"), []),
            "tables": _parse_json_or_python(d.get("tables"), []),
            "math": _parse_json_or_python(d.get("math"), []),
            "paragraphs_split": _parse_text_list(d.get("paragraphs")),
        },
    )

    paper_sources_by_link = _build_keyed_row_cache(
        papers,
        key_fn=lambda d: _resolve_link_value(d.get("paper_link")),
        value_builder=lambda d: {
            "links": _parse_json_or_python(d.get("links"), []),
            "emails": _parse_json_or_python(d.get("emails"), []),
            "codes": _parse_json_or_python(d.get("codes"), []),
            "tables": _parse_json_or_python(d.get("tables"), []),
            "math": _parse_json_or_python(d.get("math"), []),
            "paragraphs_split": _parse_text_list(d.get("paragraphs")),
        },
    )

    gh_sources_by_link = _build_keyed_row_cache(
        github,
        key_fn=lambda d: _resolve_link_value(d.get("github_link")),
        value_builder=lambda d: {
            "links": _parse_json_or_python(d.get("links"), []),
            "emails": _parse_json_or_python(d.get("emails"), []),
            "codes": _parse_json_or_python(d.get("codes"), []),
            "tables": _parse_json_or_python(d.get("tables"), []),
            "math": _parse_json_or_python(d.get("math"), []),
            "paragraphs_split": _parse_text_list(d.get("paragraphs")),
        },
    )

    model_links = _build_keyed_row_cache(
        pool,
        key_fn=lambda d: d.get("modelId", ""),
        value_builder=lambda d: {
            "paper": _resolve_link_value(d.get("primary_paper")),
            "github": _resolve_link_value(d.get("primary_github")),
            "paper_raw": _resolve_raw_link_value(d.get("primary_paper")),
            "github_raw": _resolve_raw_link_value(d.get("primary_github")),
        },
    )

    _PLACEHOLDER_SOURCE_CACHE = {
        "mc_sources": mc_sources,
        "paper_sources_by_link": paper_sources_by_link,
        "gh_sources_by_link": gh_sources_by_link,
        "model_links": model_links,
    }
    return _PLACEHOLDER_SOURCE_CACHE


def _build_csv_backed_replacements_for_model(model_id: str) -> dict[str, str]:
    cache = _load_placeholder_source_cache()
    model_links = cache["model_links"].get(model_id, {})

    replacements: dict[str, str] = {}

    mc_payload = cache["mc_sources"].get(model_id, {})
    replacements.update(_build_placeholder_map_for_source("MC", mc_payload))

    paper_link = str(model_links.get("paper", ""))
    paper_payload = cache["paper_sources_by_link"].get(paper_link, {})
    replacements.update(_build_placeholder_map_for_source("RP", paper_payload))

    gh_link = str(model_links.get("github", ""))
    gh_payload = cache["gh_sources_by_link"].get(gh_link, {})
    replacements.update(_build_placeholder_map_for_source("GH", gh_payload))

    return replacements


def _build_csv_backed_source_payloads_for_model(model_id: str) -> dict[str, dict[str, Any]]:
    cache = _load_placeholder_source_cache()
    model_links = cache["model_links"].get(model_id, {})

    paper_link = str(model_links.get("paper", ""))
    gh_link = str(model_links.get("github", ""))

    return {
        "MC": cache["mc_sources"].get(model_id, {}),
        "RP": cache["paper_sources_by_link"].get(paper_link, {}),
        "GH": cache["gh_sources_by_link"].get(gh_link, {}),
    }


def _build_csv_backed_artifact_chunks_for_model(model_id: str) -> dict[str, dict[str, Any]]:
    cache = _load_placeholder_source_cache()
    model_links = cache["model_links"].get(model_id, {})

    paper_link_norm = str(model_links.get("paper", ""))
    github_link_norm = str(model_links.get("github", ""))

    mc_payload = cache["mc_sources"].get(model_id, {})
    pp_payload = cache["paper_sources_by_link"].get(paper_link_norm, {})
    gh_payload = cache["gh_sources_by_link"].get(github_link_norm, {})

    if not pp_payload.get("paragraphs_split") and paper_link_norm:
        try:
            retrieval = retrieve_model_artifacts(model_id)
            if retrieval.paper_content:
                paper = preprocess_paper(
                    retrieval.paper_content,
                    paper_json=retrieval.paper_json,
                    declared_language=retrieval.card_data.get("language"),
                    summarize_tables=False,
                )
                pp_payload = {
                    "links": list(paper.links_map.values()),
                    "paragraphs_split": list(paper.paragraphs_split),
                }
        except Exception:
            pass

    return {
        "model_card": {
            "source_link": None,
            "links": [str(x) for x in (mc_payload.get("links") or [])],
            "paragraphs_split": [str(x) for x in (mc_payload.get("paragraphs_split") or [])],
        },
        "paper": {
            "source_link": str(model_links.get("paper_raw", "") or model_links.get("paper", "") or "") or None,
            "links": [str(x) for x in (pp_payload.get("links") or [])],
            "paragraphs_split": [str(x) for x in (pp_payload.get("paragraphs_split") or [])],
        },
        "github_readme": {
            "source_link": str(model_links.get("github_raw", "") or model_links.get("github", "") or "") or None,
            "links": [str(x) for x in (gh_payload.get("links") or [])],
            "paragraphs_split": [str(x) for x in (gh_payload.get("paragraphs_split") or [])],
        },
    }


def _placeholder_list_key(token_type: str) -> str:
    return {
        "LINK": "links",
        "EMAIL": "emails",
        "CODE": "codes",
        "TABLE": "tables",
        "MATH": "math",
    }.get(token_type, "")


def _source_priority_for_section(*, section_name: str, token_type: str) -> list[str]:
    token_type = token_type.upper()
    sec = (section_name or "").strip().lower()

    if token_type == "MATH":
        return ["RP", "GH", "MC"]

    if "how to use" in sec:
        return ["GH", "MC", "RP"]

    if "contact" in sec:
        return ["GH", "MC", "RP"]

    if any(k in sec for k in ("training", "evaluation", "model details", "environmental")):
        return ["RP", "GH", "MC"]

    return ["MC", "GH", "RP"]


def _resolve_placeholder_value(
    *,
    source_payloads: dict[str, dict[str, Any]],
    section_name: str,
    explicit_prefix: str | None,
    index: int,
    token_type: str,
) -> str | None:
    key = _placeholder_list_key(token_type)
    if not key:
        return None

    priority = _source_priority_for_section(section_name=section_name, token_type=token_type)
    if explicit_prefix:
        priority = [explicit_prefix] + [p for p in priority if p != explicit_prefix]

    for src in priority:
        payload = source_payloads.get(src, {})
        vals = payload.get(key) if isinstance(payload, dict) else None
        if not isinstance(vals, list):
            continue
        if 0 <= index < len(vals):
            return _stringify_structured_node(vals[index])

    return None


def _resolve_remaining_placeholders_for_section(
    text: str,
    *,
    section_name: str,
    source_payloads: dict[str, dict[str, Any]],
) -> str:
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        prefix = match.group(1).upper() if match.group(1) else None
        idx = int(match.group(2))
        token_type = match.group(3).upper()

        resolved = _resolve_placeholder_value(
            source_payloads=source_payloads,
            section_name=section_name,
            explicit_prefix=prefix,
            index=idx,
            token_type=token_type,
        )
        if resolved is None:
            return match.group(0)

        if token_type == "CODE" and "```" in resolved:
            if _is_inside_fenced_code_block(text, match.start()):
                return _extract_code_body(resolved)
            return f"\n\n{resolved.strip()}\n\n"

        return resolved

    resolved = _SECTION_AWARE_PLACEHOLDER_PATTERN.sub(repl, text)
    resolved = _SECTION_AWARE_DOLLAR_PLACEHOLDER_PATTERN.sub(repl, resolved)
    return resolved


def _normalize_formula_text(text: str) -> str:
    if not text:
        return text

    out = text
    out = out.replace("∑i=15", "\\sum_{i=1}^{5}")
    out = re.sub(r"\bsi\b", "s_i", out)
    out = out.replace(
        "\\frac{\\sum_{i=1}^{5} s_i · 𝕀(s_i ≠ 0)}{\\sum_{i=1}^{5} 𝕀(s_i ≠ 0)}",
        "\\frac{\\sum_{i=1}^{5} s_i \\cdot \\mathbb{I}(s_i \\neq 0)}{\\sum_{i=1}^{5} \\mathbb{I}(s_i \\neq 0)}",
    )
    return out


def _load_repro_score_cache() -> dict[str, float]:
    global _REPRO_SCORE_CACHE
    if _REPRO_SCORE_CACHE is not None:
        return _REPRO_SCORE_CACHE

    reports = _load_repro_report_cache()
    scores: dict[str, float] = {}
    for mid, emc_eval in reports.items():
        raw = emc_eval.get("total_score")
        try:
            scores[mid] = float(raw)
        except Exception:  # noqa: BLE001
            continue

    _REPRO_SCORE_CACHE = scores
    return scores


def _load_repro_report_cache() -> dict[str, dict[str, Any]]:
    global _REPRO_REPORT_CACHE, _REPRO_REPORT_CACHE_MTIME, _REPRO_SCORE_CACHE

    current_mtime: float | None = None
    if _SAMPLE_REPRO_REPORTS_PATH.exists():
        try:
            current_mtime = _SAMPLE_REPRO_REPORTS_PATH.stat().st_mtime
        except OSError:
            current_mtime = None

    if _REPRO_REPORT_CACHE is not None and _REPRO_REPORT_CACHE_MTIME == current_mtime:
        return _REPRO_REPORT_CACHE

    reports: dict[str, dict[str, Any]] = {}
    if not _SAMPLE_REPRO_REPORTS_PATH.exists():
        _REPRO_REPORT_CACHE = reports
        _REPRO_REPORT_CACHE_MTIME = current_mtime
        _REPRO_SCORE_CACHE = None
        return reports

    with _SAMPLE_REPRO_REPORTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue

            mid = str(row.get("modelId", "")).strip()
            if not mid:
                continue

            emc_eval = row.get("emc_eval") if isinstance(row.get("emc_eval"), dict) else {}
            if not emc_eval:
                continue

            if "section_scores" not in emc_eval and isinstance(emc_eval.get("scores_by_section"), dict):
                emc_eval["section_scores"] = emc_eval.get("scores_by_section")

            if "success" not in emc_eval:
                emc_eval["success"] = True

            reports[mid] = emc_eval

    _REPRO_REPORT_CACHE = reports
    _REPRO_REPORT_CACHE_MTIME = current_mtime
    _REPRO_SCORE_CACHE = None
    return reports