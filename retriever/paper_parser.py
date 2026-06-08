"""Paper PDF parser — download and extract text content from research papers.

Supports two backends:

* **``pymupdf``** (default) — fast, CPU-only, <1 s per paper.
  Best for the web app and quick local use.

* **``marker``** — high-quality extraction with equation/table support,
  requires GPU + heavy dependencies. Best for batch experiments where
  output quality matters for publication.

Usage::

    # Fast (web app)
    result = parse_paper_pdf(url)

    # High-quality (experiments)
    result = parse_paper_pdf(url, backend="marker")
"""

from __future__ import annotations

import gc
import json
import tempfile
from dataclasses import dataclass
from typing import Literal

import requests

# Lazy-loaded globals — heavy imports deferred until first call.
_marker_converter = None


@dataclass
class ParsedPaper:
    """Extracted content from a research paper PDF."""

    content: str = ""
    json_content: str = ""  # Raw marker JSON (empty when pymupdf backend is used)
    success: bool = False


def parse_paper_pdf(
    pdf_url: str,
    *,
    backend: Literal["pymupdf", "marker"] = "pymupdf",
    timeout: int = 30,
) -> ParsedPaper:
    """Download a PDF from *pdf_url* and extract its text content.

    Parameters
    ----------
    pdf_url : str
        Direct URL to a PDF file (e.g. ``https://arxiv.org/pdf/2301.00001.pdf``).
    backend : ``"pymupdf"`` | ``"marker"``
        ``"pymupdf"`` — fast, CPU-only (default, used by web app).
        ``"marker"``  — high-quality, GPU (used by batch experiments).
    timeout : int
        HTTP request timeout in seconds.
    """
    # --- Download PDF -------------------------------------------------------
    try:
        resp = requests.get(pdf_url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return ParsedPaper()

    # --- Parse --------------------------------------------------------------
    if backend == "marker":
        return _parse_with_marker(resp.content)
    else:
        return _parse_with_pymupdf(resp.content)


# ---------------------------------------------------------------------------
# Backend: PyMuPDF (fast, CPU-only)
# ---------------------------------------------------------------------------
def _parse_with_pymupdf(pdf_bytes: bytes) -> ParsedPaper:
    """Extract text from PDF bytes using PyMuPDF (fitz)."""
    try:
        import pymupdf

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()

        content = "\n\n".join(pages)
        return ParsedPaper(content=content, success=True)

    except Exception:
        return ParsedPaper()


# ---------------------------------------------------------------------------
# Backend: Marker (high-quality, GPU)
# ---------------------------------------------------------------------------
def _get_marker_converter():
    """Create (or return cached) marker PDF converter."""
    global _marker_converter
    if _marker_converter is not None:
        return _marker_converter

    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config = {
        "output_format": "json",
        "disable_image_extraction": True,
    }
    config_parser = ConfigParser(config)

    _marker_converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    return _marker_converter


def _parse_with_marker(pdf_bytes: bytes) -> ParsedPaper:
    """Extract text from PDF bytes using marker (high-quality).

    Returns both plain text (``content``) and raw JSON (``json_content``)
    when marker produces structured output.  The JSON is consumed by the
    preprocessor for section / table / equation extraction.
    """
    try:
        converter = _get_marker_converter()

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()

            rendered = converter(tmp.name)

            from marker.output import text_from_rendered
            paper_content = text_from_rendered(rendered)

            # Capture structured JSON for downstream preprocessing.
            json_content = ""
            try:
                if hasattr(rendered, "model_dump_json"):
                    json_content = rendered.model_dump_json()
                elif hasattr(rendered, "model_dump"):
                    json_content = json.dumps(rendered.model_dump())
                elif isinstance(rendered, dict):
                    json_content = json.dumps(rendered)
            except Exception:  # noqa: BLE001
                pass

        return ParsedPaper(content=paper_content, json_content=json_content, success=True)

    except Exception:
        return ParsedPaper()

    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
