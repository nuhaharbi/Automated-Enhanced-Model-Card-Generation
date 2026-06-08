"""Data models for paragraph-level classification output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ParagraphPrediction:
    """Single paragraph classification output."""

    paragraph: str
    predicted_label: str
    confidence: float
    top2_label: str | None = None
    top2_confidence: float | None = None
    source: str = ""
    index: int = 0
    applied_rerank_rule: bool = False


@dataclass
class ArtifactClassification:
    """Predictions for one artifact source (paper/model-card/github)."""

    source: str
    predictions: list[ParagraphPrediction] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Full paragraph-level classification output for one model."""

    model_id: str
    labels: list[str] = field(default_factory=list)
    paper: ArtifactClassification = field(default_factory=lambda: ArtifactClassification(source="paper"))
    model_card: ArtifactClassification = field(
        default_factory=lambda: ArtifactClassification(source="model_card")
    )
    github_readme: ArtifactClassification = field(
        default_factory=lambda: ArtifactClassification(source="github_readme")
    )
    tuning_status: dict[str, Any] = field(default_factory=dict)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dictionary."""
        return asdict(self)
