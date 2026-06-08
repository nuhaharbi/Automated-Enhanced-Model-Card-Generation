"""Data models for structured retrieval output."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RetrievalResult:
    """Complete retrieval output for a single HuggingFace model."""

    model_id: str
    author: str | None = None

    # --- Model card & metadata ---
    model_card: str = ""
    tags: list[str] = field(default_factory=list)
    card_data: dict[str, Any] = field(default_factory=dict)
    downloads: int = 0
    likes: int = 0
    trending_score: float | None = None

    # --- Paper links (produced by Algorithm 1) ---
    primary_paper_url: str | None = None
    secondary_paper_urls: list[str] = field(default_factory=list)

    # --- GitHub links (produced by Algorithm 2) ---
    primary_github_url: str | None = None
    secondary_github_urls: list[str] = field(default_factory=list)

    # --- GitHub repo content (fetched via GitHub API) ---
    github_readme: str = ""
    github_files: list[str] = field(default_factory=list)

    # --- Paper content (parsed from PDF) ---
    paper_content: str = ""
    paper_json: str = ""  # Raw marker JSON for structured preprocessing

    # --- HF repo metadata ---
    model_files: list[str] = field(default_factory=list)
    model_discussions: list[dict[str, Any]] = field(default_factory=list)

    # --- Pipeline task ---
    task: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return asdict(self)
