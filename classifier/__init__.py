"""Classifier component (Component 3).

This package is intentionally paragraph-only for the first integration phase.
"""

from classifier.core import classify_paragraphs, classify_preprocessed_paragraphs
from classifier.models import (
    ArtifactClassification,
    ClassificationResult,
    ParagraphPrediction,
)

__all__ = [
    "ArtifactClassification",
    "ClassificationResult",
    "ParagraphPrediction",
    "classify_paragraphs",
    "classify_preprocessed_paragraphs",
]
