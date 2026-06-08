"""Data models for Enhanced Model Card generation output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnhancedModelCardResult:
    """Generated Enhanced Model Card sections and metadata."""

    model_id: str
    model_name: str
    sections: dict[str, str] = field(default_factory=dict)
    full_markdown: str = ""
    success: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
