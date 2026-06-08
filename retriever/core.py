"""Core orchestrator — single entry point for retrieving all model artifacts."""

from __future__ import annotations

from retriever.github_client import fetch_github_content
from retriever.hf_client import fetch_model_card, fetch_model_metadata
from retriever.link_extractor import LinkExtractor
from retriever.models import RetrievalResult
from retriever.paper_parser import parse_paper_pdf


def retrieve_model_artifacts(model_id: str) -> RetrievalResult:
    """Retrieve all artifacts for a single HuggingFace model.

    Pipeline:
        1. Fetch raw model card (README.md)
        2. Fetch metadata via HF Hub API (tags, files, discussions, …)
        3. Extract & rank paper + GitHub links (Algorithms 1 & 2)
        4. Return a structured ``RetrievalResult``

    Parameters
    ----------
    model_id : str
        HuggingFace repo identifier, e.g. ``"meta-llama/Llama-2-7b"``.
    """
    # --- Step 1: Model card ------------------------------------------------
    model_card = fetch_model_card(model_id)
    if model_card is None:
        return RetrievalResult(model_id=model_id)

    # --- Step 2: Metadata --------------------------------------------------
    meta = fetch_model_metadata(model_id)
    tags = meta.get("tags", [])

    # --- Step 3: Link extraction -------------------------------------------
    extractor = LinkExtractor(model_id)
    primary_paper, secondary_papers, primary_github, secondary_github = (
        extractor.extract_all(model_card)
    )

    # --- Step 4: Fetch GitHub repo content ----------------------------------
    github_readme = ""
    github_files: list[str] = []
    if primary_github:
        gh = fetch_github_content(primary_github)
        github_readme = gh.readme
        github_files = gh.files

    # --- Step 5: Parse primary paper PDF ------------------------------------
    paper_content = ""
    paper_json = ""
    if primary_paper:
        parsed = parse_paper_pdf(primary_paper)
        if parsed.success:
            paper_content = parsed.content
            paper_json = parsed.json_content

    # --- Step 6: Assemble result -------------------------------------------
    return RetrievalResult(
        model_id=model_id,
        author=meta.get("author"),
        model_card=model_card,
        tags=tags,
        card_data=meta.get("card_data", {}),
        downloads=meta.get("downloads", 0),
        likes=meta.get("likes", 0),
        trending_score=meta.get("trending_score"),
        primary_paper_url=primary_paper,
        secondary_paper_urls=secondary_papers,
        primary_github_url=primary_github,
        secondary_github_urls=secondary_github,
        github_readme=github_readme,
        github_files=github_files,
        paper_content=paper_content,
        paper_json=paper_json,
        model_files=meta.get("model_files", []),
        model_discussions=meta.get("model_discussions", []),
    )
