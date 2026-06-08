"""HuggingFace API client — fetch model metadata and card content."""

from __future__ import annotations

from typing import Any

from huggingface_hub import HfApi, ModelCard

_api = HfApi()


def fetch_model_card(repo_id: str, *, timeout: int = 10) -> str | None:
    """Load the model card for *repo_id* via ``ModelCard.load()``.

    Returns the card text, or ``None`` if it cannot be fetched.
    """
    try:
        card = ModelCard.load(repo_id)
        return str(card)
    except Exception:
        return None


def fetch_model_metadata(repo_id: str) -> dict[str, Any]:
    """Return key metadata fields for *repo_id* via the HF Hub API.

    Returns a dict with: author, tags, card_data, downloads, likes,
    trending_score, model_files, model_discussions.
    """
    try:
        info = _api.model_info(repo_id)
    except Exception:
        return {}

    # Discussions
    try:
        discussions = [
            {
                "title": d.title,
                "num": d.num,
                "author": d.author,
                "status": d.status,
                "is_pull_request": d.is_pull_request,
            }
            for d in _api.get_repo_discussions(repo_id=repo_id)
        ]
    except Exception:
        discussions = []

    # File listing
    try:
        model_files = list(_api.list_repo_files(repo_id=repo_id))
    except Exception:
        model_files = []

    card_data = info.card_data.to_dict() if info.card_data is not None else {}

    return {
        "author": info.author,
        "tags": list(info.tags) if info.tags else [],
        "card_data": card_data,
        "downloads": info.downloads,
        "likes": info.likes,
        "trending_score": getattr(info, "trending_score", None),
        "model_files": model_files,
        "model_discussions": discussions,
    }
