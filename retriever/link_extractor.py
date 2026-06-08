"""Link extraction and ranking from HuggingFace model cards.

This module implements the collection flowchart described in the publication:

1. **Paper gate** – A model is only considered if its model card contains
   at least one research-paper link (arXiv URL, BibTeX entry, or other
   academic URL).  Models without any paper link are excluded.

2. **ArXiv title / abstract check** – When arXiv links exist, each paper's
   title and abstract are fetched from the arXiv API and checked for the
   model name.  A single match is accepted directly as the *primary paper*;
   multiple matches are disambiguated by Algorithm 1.

3. **Algorithm 1 – Paper ranking** (`_rank_papers`):
   When no title/abstract match is found (or there are no arXiv links),
   candidates are ranked by
   *in-text frequency × 2 + inverse-position (continuous decay) + source bonus*
   in the model card.
   Source bonuses: explicit label (+10), BibTeX (+5).
   An *explicit label* is a line in the card body starting with
   ``Paper:`` or ``Link to paper:``.

4. **GitHub extraction & Algorithm 2 – GitHub filtering** (`_filter_github`):
   GitHub links in the model card are scored by token-overlap
   (+3 per shared word token between the GitHub ``org/repo`` path and
   the model name), explicit-label bonus (+5 when the link appears on
   a line labelled ``Code:``, ``[GitHub]``, ``Repository:``, etc.),
   and a first-link position bonus (+2).

5. **ArXiv abstract fallback** – If no GitHub link is found in the model
   card (or Algorithm 2 yields zero overlap), and the primary paper is
   from arXiv, the paper's abstract is searched for GitHub links as a
   last resort.

6. **Exclusion** – If no GitHub link is found after all steps, the model
   is excluded regardless of the primary paper source.

Note: HuggingFace ``arxiv:`` tags are **not** examined for arXiv IDs
because, per HF documentation, these tags are generated automatically
from the model card content and would be redundant.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import bibtexparser

from retriever.arxiv_client import fetch_arxiv_title_and_abstract, format_arxiv_pdf_url

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
URL_REGEX = (
    r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}"
    r"\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)"
)

GITHUB_REGEX = (
    r"(?:(?:git|ssh|http(?:s)?)|(?:git@[\w\.]+))"
    r"(?::(?://)?)github\.com\/(?:[\w\.@\:/\-~]+)(?:/)?"
)

# Simpler pattern that also matches bare ``github.com/org/repo`` in abstracts
GITHUB_URL_SIMPLE = r"(?:https?://)?github\.com/[\w.\-]+/[\w.\-]+"


# File extensions that indicate a link to an asset, not a repository.
_ASSET_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".tgz",
    ".mp4", ".mp3", ".wav", ".avi",
    ".csv", ".json", ".jsonl", ".yaml", ".yml", ".txt",
}

# Lines that explicitly label a paper (case-insensitive).  Matches:
#   Paper: <url>   |   **Paper**: <url>   |   Link to paper: <url>
#   Link to the paper: <url>
#   [Paper: Title](url)  |  - **Paper**: <url>  (Markdown link / list item)
# NOTE: "Original paper" is deliberately excluded — it typically refers
# to a predecessor / base model's paper, not this model's own paper.
_PAPER_LABEL_RE = re.compile(
    r"^(?:[-*+]\s+)?"          # optional list marker  (- / * / +)
    r"\[?\s*"                   # optional Markdown-link opening bracket
    r"\*{0,2}"
    r"(?:paper|link to (?:the )?paper)"
    r"\*{0,2}"
    r"\s*:",
    re.IGNORECASE,
)

# Lines that explicitly label a GitHub link (case-insensitive).
# Checked per-line: if a line contains a GitHub URL AND matches this
# pattern, the URL receives a label bonus in Algorithm 2.
_GITHUB_LABEL_RE = re.compile(
    r"(?:"
    r"\[(?:github|code|repo(?:sitory)?|source(?:\s*code)?|implementation)\]"
    r"|"
    r"\*{0,2}(?:github(?:\s*(?:repo(?:sitory)?|link))?|code|repo(?:sitory)?"
    r"|source(?:\s*code)?|implementation)\*{0,2}\s*:"
    r"|"
    r"(?:official|original)\s+(?:code|implementation|repo(?:sitory)?)"
    r"|"
    r"(?:code|source(?:\s*code)?)\s+(?:is\s+)?(?:available|released)\s+(?:at|on|in)"
    r")",
    re.IGNORECASE,
)


class LinkExtractor:
    """Extract, rank and filter paper / GitHub links from a model card.

    Parameters
    ----------
    repo_id : str
        Full HuggingFace repo identifier, e.g. ``"meta-llama/Llama-2-7b"``.
    paper_freq_weight : int
        Multiplier for in-text frequency in Algorithm 1 (default ``2``).
    paper_label_bonus : int
        Bonus for papers on explicitly labelled lines in Algorithm 1
        (default ``5``).  Empirically determined: the minimum value
        that preserves all primary-paper selections on 3,366 models.
    paper_bibtex_bonus : int
        Bonus for papers found in BibTeX blocks in Algorithm 1
        (default ``5``).
    github_token_bonus : int
        Per-token bonus for model-name overlap in Algorithm 2
        (default ``3``).
    github_label_bonus : int
        Bonus for GitHub links on explicitly labelled lines in Algorithm 2
        (default ``5``).  Empirically determined: the minimum value
        that preserves all primary-GitHub selections on 3,366 models.
    github_position_bonus : int
        Bonus for the first GitHub link in card order in Algorithm 2
        (default ``2``).
    github_min_score : int
        Minimum Algorithm 2 score before falling back to the
        dominant-repo heuristic (default ``3``).
    """

    def __init__(
        self,
        repo_id: str,
        *,
        paper_freq_weight: int = 2,
        paper_label_bonus: int = 5,
        paper_bibtex_bonus: int = 5,
        github_token_bonus: int = 3,
        github_label_bonus: int = 5,
        github_position_bonus: int = 2,
        github_min_score: int = 3,
    ) -> None:
        self.repo_id = repo_id
        self.model_id_lower = repo_id.lower()  # e.g. "alibaba-damo/mgp-str-base"

        # Algorithm 1 (paper ranking) parameters
        self.paper_freq_weight = paper_freq_weight
        self.paper_label_bonus = paper_label_bonus
        self.paper_bibtex_bonus = paper_bibtex_bonus

        # Algorithm 2 (GitHub filtering) parameters
        self.github_token_bonus = github_token_bonus
        self.github_label_bonus = github_label_bonus
        self.github_position_bonus = github_position_bonus
        self.github_min_score = github_min_score

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def extract_all(
        self,
        content: str,
        *,
        skip_arxiv_api: bool = False,
    ) -> tuple[
        Optional[str],      # primary paper URL
        list[str],           # secondary paper URLs
        Optional[str],       # primary GitHub URL
        list[str],           # secondary GitHub URLs
    ]:
        """Orchestrate full extraction from model card *content*.

        The flow mirrors the corrected collection flowchart:

        1. Collect all paper references (arXiv IDs from card content and
           BibTeX + other academic URLs).  If none exist → exclude.
        2. When arXiv IDs are present, check each paper's title & abstract
           for the model name; a direct match becomes the primary paper.
        3. If no direct match (or no arXiv IDs) → **Algorithm 1** ranks
           candidates by frequency + position.
        4. Search the model card for GitHub links.

           * Found → **Algorithm 2** picks the best match by word overlap.
           * Algorithm 2 succeeds → done (include).

        5. *Fallback*: if no GitHub was found (or Algorithm 2 scored 0)
           and the primary paper is from arXiv, search the paper's abstract
           for GitHub links and re-run Algorithm 2.
        6. If still no GitHub → exclude.

        Parameters
        ----------
        skip_arxiv_api : bool
            When ``True``, do **not** call the arXiv API for title/abstract
            checking (Step 2) or abstract fallback (Step 5).  Useful in batch
            re-evaluation mode to avoid thousands of slow API calls — the
            primary paper will be selected solely by Algorithm 1.

        Returns
        -------
        (primary_paper_url, secondary_paper_urls,
         primary_github_url, secondary_github_urls)
        """
        # === Step 1: Collect all paper candidates ===========================
        content_ids = self._extract_arxiv_ids_from_urls(content)
        bibtex_ids = self._extract_arxiv_ids_from_bibtex(content)
        all_arxiv_ids = list(dict.fromkeys(content_ids + bibtex_ids))

        other_urls = self._extract_other_paper_urls(content)

        # [The repo has research paper(s) links?]
        if not all_arxiv_ids and not other_urls:
            return None, [], None, []          # No papers → EXCLUDE

        # === Step 2: Build candidates & check ArXiv title/abstract ==========
        candidates: list[tuple[str, str]] = []
        matches: list[tuple[str, str]] = []
        arxiv_abstracts: dict[str, str] = {}   # arxiv_id → title+abstract

        for arxiv_id in all_arxiv_ids:
            candidate = format_arxiv_pdf_url(arxiv_id)
            candidates.append(candidate)              # always keep (Fix 3)

            if not skip_arxiv_api:
                title_abstract = fetch_arxiv_title_and_abstract(arxiv_id)
                if title_abstract:
                    arxiv_abstracts[arxiv_id] = title_abstract
                    ta_lower = title_abstract.lower()
                    name_stem = self.model_id_lower.rsplit("/", 1)[-1].split("-")[0]
                    if self.model_id_lower.rsplit("/", 1)[-1] in ta_lower:
                        matches.append(candidate)
                    elif len(name_stem) >= 3 and name_stem in ta_lower:
                        matches.append(candidate)      # stem guard (Fix 1)

        candidates += other_urls

        if not candidates:
            return None, [], None, []          # No valid candidates → EXCLUDE

        # === Step 3: Determine primary paper ================================
        # [Is there ArXiv links?] → Yes → [Check paper(s) title & abstract]
        if all_arxiv_ids and len(matches) == 1:
            # Primary Paper found directly (single match)
            primary_id = matches[0][0]
        elif len(candidates) > 1:
            # Multiple matches / no match → Algorithm 1 ranks ALL (Fix 4)
            card_body = self._strip_yaml_front_matter(content)
            labeled_ids = self._extract_labeled_paper_refs(card_body)
            primary_id = (
                self._rank_papers(
                    [c[0] for c in candidates], card_body,
                    bibtex_ids=set(bibtex_ids),
                    labeled_ids=labeled_ids,
                )
                or candidates[0][0]                    # safety fallback (Fix 2)
            )
        else:
            primary_id = candidates[0][0]

        primary_is_arxiv = primary_id in arxiv_abstracts
        primary_url = next(c[1] for c in candidates if c[0] == primary_id)
        secondary_urls = list({c[1] for c in candidates if c[1] != primary_url})

        # === Step 4: GitHub links from model card ===========================
        all_github_links = list(dict.fromkeys(       # deduplicate (Fix 6)
            self._extract_github_links(content)
        ))
        best_github: Optional[str] = None
        other_github: list[str] = []

        if all_github_links:
            # [The repo has GitHub links?] → Yes → Algorithm 2
            best_github, other_github = self._filter_github(all_github_links, content)

        if best_github:
            # [Primary GitHub found?] → Yes → INCLUDE
            return primary_url, secondary_urls, best_github, other_github

        # === Step 5: Fallback — search ArXiv abstract for GitHub links ======
        # [The primary paper from ArXiv?]
        if not skip_arxiv_api and primary_is_arxiv:
            # → Yes → [Check primary paper abstract]
            abstract_text = arxiv_abstracts.get(primary_id, "")
            abstract_github = re.findall(GITHUB_URL_SIMPLE, abstract_text)
            abstract_github = [u.rstrip(".,;:!?)\"'") for u in abstract_github]
            if abstract_github:
                best_github, other_github = self._filter_github(abstract_github)
                if best_github:
                    # [Primary GitHub found?] → Yes → INCLUDE
                    return primary_url, secondary_urls, best_github, other_github

        # No GitHub found at all → return papers only
        # (batch collector treats this as EXCLUDE; web tool still shows papers)
        return primary_url, secondary_urls, None, []

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def _extract_github_links(self, content: str) -> list[str]:
        """Extract GitHub URLs from *content*.

        URLs that point to images or other asset files (e.g. ``.png``,
        ``.jpg``, ``.pdf``) are silently dropped.  Links whose path
        segment is a GitHub UI page (``/blob/``, ``/raw/``,
        ``/stargazers``, ``/issues``, ``/releases``, ``/assets``,
        ``/commit``, etc.) are truncated to the repo root so we keep
        the repo but discard the non-code portion.
        """
        github_links = re.findall(GITHUB_REGEX, content)

        # Strip asset / image URLs, org-only links; truncate deep paths to repo root
        filtered: list[str] = []
        for link in github_links:
            # Strip trailing sentence punctuation captured by the regex
            link = link.rstrip(".,;:!?)\"'")
            low = link.lower().rstrip("/")
            # Drop links ending in known asset extensions
            if any(low.endswith(ext) for ext in _ASSET_EXTENSIONS):
                continue
            # Truncate /raw/ and /blob/ links to repo root (org/repo)
            # e.g. github.com/openai/CLIP/blob/main/model-card.md → github.com/openai/CLIP
            for seg in ("/blob/", "/raw/"):
                idx = low.find(seg)
                if idx != -1:
                    link = link[:idx]
                    low = link.lower().rstrip("/")
                    break
            # Truncate GitHub meta / UI pages to repo root (org/repo)
            # e.g. github.com/org/repo/stargazers → github.com/org/repo
            for seg in ("/stargazers", "/issues", "/pulls", "/wiki",
                        "/releases", "/actions", "/discussions",
                        "/assets", "/commit", "/marketplace"):
                idx = low.find(seg)
                if idx != -1:
                    link = link[:idx]
                    low = link.lower().rstrip("/")
                    break
            # Drop org/user-only links (no repo name) e.g. github.com/Stability-AI/
            path = re.sub(r"https?://github\.com/?", "", low).strip("/")
            if not path or "/" not in path:
                continue
            filtered.append(link)

        return filtered

    def _extract_arxiv_ids_from_urls(self, content: str) -> list[str]:
        """Extract arXiv IDs from ``arxiv.org`` and ``huggingface.co/papers`` URLs."""
        # arxiv.org/abs/<id>  or  arxiv.org/pdf/<id>
        arxiv_matches = re.findall(
            r"https?://arxiv\.org/(abs|pdf)/(\d+\.\d+)", content
        )
        # huggingface.co/papers/<id>  or  hf.co/papers/<id>
        hf_matches = re.findall(
            r"https?://(?:huggingface\.co|hf\.co)/papers/(\d+\.\d+)", content
        )
        return [m[1] for m in arxiv_matches] + hf_matches

    def _extract_arxiv_ids_from_bibtex(self, content: str) -> list[str]:
        """Extract arXiv IDs from BibTeX blocks embedded in the model card."""
        code_blocks = re.finditer(r"@\w+\{.*?\n\s*\}", content, re.DOTALL)
        arxiv_ids: list[str] = []

        for block in code_blocks:
            bib_str = block.group(0)
            try:
                db = bibtexparser.loads(bib_str)
                if not db.entries:
                    continue

                for entry in db.entries:
                    # bibtexparser v1.x: entries are plain dicts
                    # with lowercased keys.

                    # Check `url` field
                    url = entry.get("url", "").lower()
                    if "arxiv.org" in url:
                        match = re.search(r"arxiv\.org\/abs\/([\w.\-\/]+)", url, re.I)
                        if match:
                            arxiv_ids.append(match.group(1))

                    # Check explicit `arxiv` field
                    arxiv_field = entry.get("arxiv", "").strip()
                    if arxiv_field:
                        arxiv_ids.append(arxiv_field)

                    # Check `archivePrefix` + `eprint` combo
                    # (v1.x lowercases keys → "archiveprefix")
                    if entry.get("archiveprefix", "").lower() == "arxiv":
                        eprint = entry.get("eprint", "").strip()
                        if eprint:
                            arxiv_ids.append(eprint)

                    # Fallback: scan ALL field values for the common
                    # "arXiv preprint arXiv:XXXX.XXXXX" or bare
                    # "arXiv:XXXX.XXXXX" pattern (often in journal/note).
                    for val in entry.values():
                        if not isinstance(val, str):
                            continue
                        for m in re.finditer(r"arXiv[:\s]+(?:preprint\s+arXiv[:\s]+)?(\d{4}\.\d{4,5})", val, re.I):
                            arxiv_ids.append(m.group(1))

            except Exception:
                continue

        return arxiv_ids

    def _extract_other_paper_urls(self, content: str) -> list[tuple[str, str]]:
        """Extract non-arXiv paper URLs (ACL Anthology, CVF OpenAccess)."""
        all_urls = re.findall(URL_REGEX, content)
        papers: list[tuple[str, str]] = []

        for url in all_urls:
            url = url.rstrip("/")
            if "anthology.org" in url:
                pdf = url if url.endswith(".pdf") else url + ".pdf"
                papers.append((url, pdf))
            elif "openaccess.thecvf.com" in url:
                pdf = url.replace("/html/", "/papers/").replace(".html", ".pdf")
                papers.append((url, pdf))

        return papers

    @staticmethod
    def _extract_labeled_paper_refs(content: str) -> set[str]:
        """Return arXiv IDs **and** URLs on lines explicitly labelled as
        the paper (e.g. ``Paper:``, ``Link to paper:``).

        These are strong author-provided signals and receive a source
        bonus in :meth:`_rank_papers`.
        """
        arxiv_pat = re.compile(r"(\d{4}\.\d{4,5})")
        url_pat = re.compile(URL_REGEX)
        labeled: set[str] = set()
        for line in content.split("\n"):
            if _PAPER_LABEL_RE.match(line.strip()):
                for m in arxiv_pat.finditer(line):
                    labeled.add(m.group(1))
                # Also capture non-arXiv URLs (e.g. ACL Anthology)
                for m in url_pat.finditer(line):
                    url = m.group(0).rstrip("/")
                    if "arxiv.org" not in url.lower():
                        labeled.add(url)
        return labeled

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_yaml_front_matter(content: str) -> str:
        """Return the card body after the YAML front-matter block.

        Model cards typically start with ``---\n...metadata...\n---``.
        Position-based ranking should ignore this metadata section so that
        papers mentioned only in YAML fields (e.g. ``co2_eq_emissions``)
        do not receive an unfair position bonus.
        """
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                return content[end + 3:]
        return content

    # ------------------------------------------------------------------
    # Algorithm 1 — Paper ranking (frequency + position)
    # ------------------------------------------------------------------
    def _rank_papers(
        self,
        candidates: list[str],
        content: str,
        bibtex_ids: set[str] | None = None,
        labeled_ids: set[str] | None = None,
    ) -> str | None:
        """Rank paper candidates by *in-text frequency × 2 + position (continuous decay) + source bonus*.

        Position scoring uses a continuous decay factor ``1 − pos/len``
        so that papers appearing earlier in the card receive proportionally
        higher scores — no arbitrary threshold is needed.

        Source bonuses
        --------------
        - Explicit label (``Paper:``, ``Link to paper``) : **+5**
        - BibTeX block : **+5** – curated, intentional citation.
        """
        bibtex_ids = bibtex_ids or set()
        labeled_ids = labeled_ids or set()
        url_ranks: dict[str, int] = {}

        # Frequency-based scoring — count in-text occurrences
        unique_candidates = set(candidates)
        for c in unique_candidates:
            freq = len(re.findall(re.escape(c), content))
            url_ranks[c] = max(freq, 1) * self.paper_freq_weight

        # Source bonus — explicit "Paper:" label
        for c in unique_candidates:
            # For arXiv: labelled ID is a substring of the candidate ID.
            # For non-arXiv: labelled URL matches the candidate URL.
            for lid in labeled_ids:
                if lid in c or c in lid:
                    url_ranks[c] += self.paper_label_bonus
                    break

        # Source bonus — BibTeX
        for c in unique_candidates:
            if c in bibtex_ids:
                url_ranks[c] += self.paper_bibtex_bonus

        # Position-based scoring — continuous decay
        # Papers appearing earlier in the card receive proportionally
        # higher scores via  ordinal_rank × (1 + decay)  where
        # decay = 1 − first_position / content_length.
        # At the very start decay ≈ 1 → score ≈ 2 × ordinal;
        # at the end decay ≈ 0 → score ≈ 1 × ordinal.
        content_len = len(content) or 1

        first_occurrence: dict[str, int] = {}
        for c in unique_candidates:
            occurrences = [m.start() for m in re.finditer(re.escape(c), content)]
            if occurrences:
                first_occurrence[c] = occurrences[0]

        sorted_by_position = sorted(first_occurrence.items(), key=lambda x: x[1])
        n = len(sorted_by_position)
        for idx, (url, pos) in enumerate(sorted_by_position):
            ordinal = n - idx
            decay = 1.0 - (pos / content_len)
            url_ranks[url] += round(ordinal * (1 + decay))

        ranked = sorted(url_ranks.items(), key=lambda x: x[1], reverse=True)
        return ranked[0][0] if ranked else None

    # ------------------------------------------------------------------
    # Algorithm 2 — GitHub filtering (token-overlap + label + position)
    # ------------------------------------------------------------------
    def _filter_github(
        self,
        all_github_links: list[str],
        content: str | None = None,
    ) -> tuple[Optional[str], list[str]]:
        """Select the best GitHub link by token-overlap + label + position.

        Each candidate URL is cleaned to its ``org/repo[/subpath]``
        (stripping ``/tree/<branch>`` etc.), lowered, and compared
        against the full model ID (e.g. ``alibaba-damo/mgp-str-base``).

        Scoring has three components:

        1. **Token-overlap** – each whole token (word) shared between
           the model-ID tokens and URL-path tokens adds ``+3`` to the
           score.  This catches meaningful name matches like ``"e5"``
           or ``"llama"`` without rewarding accidental character
           overlaps.
        2. **Explicit-label bonus** – ``+4`` when the link appears on
           a line in the card that contains an explicit label such as
           ``Code:``, ``[GitHub]``, ``**Repository**:``, or
           ``official code``.
        3. **First-link position bonus** – ``+2`` for the first
           candidate (card order).  In 85 % of cards the first
           GitHub link is the primary repository.

        Ties are broken by earlier card position.  A minimum combined
        score of 3 is required; below that threshold a *dominant-repo*
        fallback is used (the repo appearing most often wins).

        Parameters
        ----------
        all_github_links : list[str]
            De-duplicated, filtered GitHub URLs in card order.
        content : str | None
            Raw model-card text, used for label detection.  May be
            ``None`` when scoring links from an arXiv abstract (Step 5
            fallback), in which case label and position bonuses are
            skipped.
        """
        model_tokens = set(self._tokenise(self.model_id_lower))

        _TOKEN_BONUS = self.github_token_bonus
        _LABEL_BONUS = self.github_label_bonus
        _POSITION_BONUS = self.github_position_bonus

        # --- detect labelled repos in the card ---------------------------
        labeled_repos: set[str] = set()
        if content:
            _gh_url_re = re.compile(
                r"https?://github\.com/[\w.\-]+/[\w.\-]+", re.I,
            )
            for line in content.split("\n"):
                urls = _gh_url_re.findall(line)
                if urls and _GITHUB_LABEL_RE.search(line):
                    for url in urls:
                        path = re.sub(
                            r"https?://github\.com/?", "", url.lower(),
                        ).strip("/")
                        parts = path.split("/")
                        if len(parts) >= 2:
                            labeled_repos.add(f"{parts[0]}/{parts[1]}")

        # --- score each candidate ----------------------------------------
        scores: list[int] = []
        for i, link in enumerate(all_github_links):
            # Strip scheme + domain → keep "org/repo[/extras]"
            cleaned = re.sub(r"https?://github\.com/", "", link).lower()
            parts = cleaned.strip("/").split("/")
            repo_path = "/".join(parts[:2]) if len(parts) >= 2 else cleaned
            # Build full cleaned path including subdirectories
            url_path = repo_path
            if len(parts) > 2:
                extra = parts[2:]
                if extra and extra[0] in ("tree", "blob", "raw"):
                    extra = extra[2:]       # skip segment type + branch
                if extra:
                    url_path = repo_path + "/" + "/".join(extra)

            # 1. Token overlap
            url_tokens = set(self._tokenise(url_path))
            token_score = len(model_tokens & url_tokens) * _TOKEN_BONUS

            # 2. Label bonus
            label_score = _LABEL_BONUS if repo_path in labeled_repos else 0

            # 3. Position bonus (first link in card)
            position_score = _POSITION_BONUS if i == 0 else 0

            scores.append(token_score + label_score + position_score)

        # Best = highest score; ties broken by earlier position (lower idx)
        best_idx = 0
        for idx in range(1, len(scores)):
            if scores[idx] > scores[best_idx]:
                best_idx = idx

        # Require a minimum combined score to avoid noise
        if scores[best_idx] < self.github_min_score:
            # Fallback: if one repo dominates (majority of links), pick it
            # even though the name tokens don't overlap
            # (e.g. BAAI/bge-* → FlagOpen/FlagEmbedding).
            repo_for_link: list[str | None] = []
            for link in all_github_links:
                cleaned = re.sub(r"https?://github\.com/", "", link).lower()
                parts = cleaned.strip("/").split("/")
                if len(parts) >= 2:
                    repo_for_link.append(f"{parts[0]}/{parts[1]}")
                else:
                    repo_for_link.append(None)
            repo_counts = Counter(r for r in repo_for_link if r is not None)
            if repo_counts:
                dominant_repo, dominant_count = repo_counts.most_common(1)[0]
                if dominant_count > len(all_github_links) / 2:
                    dom_links = [
                        l for l, r in zip(all_github_links, repo_for_link)
                        if r == dominant_repo
                    ]
                    shortest = min(dom_links, key=len)
                    others = [l for l in all_github_links if l != shortest]
                    return shortest, others
            return None, list(all_github_links)

        primary = all_github_links[best_idx]
        others = [l for i, l in enumerate(all_github_links) if i != best_idx]
        return primary, others

    # ------------------------------------------------------------------
    # Shared text tokeniser
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Split a model ID or URL path into normalised word tokens.

        Handles camelCase, kebab-case, snake_case, and embedded numbers
        (e.g. ``"Llama-3-7B"`` → ``["llama", "3", "7b"]``).
        """
        segments = re.split(r"[-_./]", text)
        tokens: list[str] = []
        for seg in segments:
            sub_tokens = re.findall(
                r"""
                \d+[a-zA-Z]+(?:[\d.]*)  |  # e.g. 7b, 7B, 7B2.0
                [A-Z]+(?=[A-Z][a-z])     |  # e.g. 'X' in 'EraX'
                [A-Z]?[a-z]+(?:[\d.]*)   |  # e.g. 'Preview', 'v2.0'
                [A-Z]+(?:[\d.]*)         |  # e.g. 'VL', 'V2.0'
                \d+                         # plain numbers
                """,
                seg,
                re.VERBOSE,
            )
            tokens.extend(sub_tokens if sub_tokens else [seg])
        return [t.lower() for t in tokens]
