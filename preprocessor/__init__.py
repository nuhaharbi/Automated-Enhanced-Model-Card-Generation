"""Preprocessor package — structured preprocessing of model artifacts.

Implements paper, model-card, and GitHub README preprocessing (Component 2).
"""

from preprocessor.core import preprocess_markdown, preprocess_paper
from preprocessor.models import PreprocessedMarkdown, PreprocessedPaper

__all__ = [
    "preprocess_paper",
    "preprocess_markdown",
    "PreprocessedPaper",
    "PreprocessedMarkdown",
]
