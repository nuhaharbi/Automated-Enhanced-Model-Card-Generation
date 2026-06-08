"""ArXiv API utilities — fetch paper metadata and build PDF URLs."""

from __future__ import annotations

import ctypes
import functools
import threading
import arxiv

# Shared client — reused across calls to respect rate limits.
_client = arxiv.Client(page_size=1, delay_seconds=1.0, num_retries=1)


class _ArxivTimeout(Exception):
    """Raised when an arXiv API call exceeds the timeout."""


def _run_with_timeout(fn, timeout: float = 15.0):
    """Run *fn* in a daemon thread with a timeout (thread-safe).

    Unlike ``signal.SIGALRM`` this works from any thread, which is
    required when called from FastAPI / uvicorn worker threads.
    """
    result: list = []
    error: list = []

    def _target():
        try:
            result.append(fn())
        except Exception as exc:  # noqa: BLE001
            error.append(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise _ArxivTimeout("ArXiv API call timed out")
    if error:
        raise error[0]
    return result[0] if result else None


@functools.lru_cache(maxsize=8192)
def fetch_arxiv_title_and_abstract(arxiv_id: str) -> str | None:
    """Return *title + abstract* for an arXiv paper, or ``None`` on failure.

    Results are cached in-memory so repeated lookups for the same paper
    across the CSV re-run cost nothing.  A 15-second timeout prevents the
    process from hanging on rate-limited responses.
    """
    def _fetch():
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(_client.results(search))
        return f"{result.title} {result.summary}"

    try:
        return _run_with_timeout(_fetch, timeout=15.0)
    except (_ArxivTimeout, StopIteration, Exception):
        return None


def format_arxiv_pdf_url(arxiv_id: str) -> tuple[str, str]:
    """Return ``(arxiv_id, pdf_url)`` for a given arXiv identifier."""
    return (arxiv_id, f"https://arxiv.org/pdf/{arxiv_id}.pdf")
