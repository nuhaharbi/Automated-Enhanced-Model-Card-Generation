"""Data models for reproducibility evaluation output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvaluationItem:
    """One scored reproducibility checklist point."""

    point_number: int
    evidence_quote: str
    reasoning: str
    score: float


@dataclass
class SectionEvaluation:
    """Evaluation output for one section checklist subset."""

    section: str
    evaluations: list[EvaluationItem] = field(default_factory=list)


@dataclass
class ReproducibilityResult:
    """Aggregated reproducibility scoring result."""

    model_id: str
    model_name: str
    scores_by_point: dict[str, float] = field(default_factory=dict)
    section_scores: dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    section_evaluations: list[SectionEvaluation] = field(default_factory=list)
    success: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dictionary."""
        return asdict(self)
