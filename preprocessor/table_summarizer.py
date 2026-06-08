"""LLM-based table summarisation for research papers and markdown files.

Generates concise natural-language descriptions of tables.  The LLM model
is **lazily loaded** on first call and cached for reuse.

Usage::

    summarizer = TableSummarizer()

    # Paper tables (marker JSON — list of block dicts)
    descriptions = summarizer.summarize(raw_tables)

    # Single markdown table (mistune AST node)
    description = summarizer.summarize_single(node)

    # Custom prompt for any table data
    description = summarizer.summarize_single(data, prompt="Describe: {table}")

The summariser is intentionally kept as a **separate class** so it can be:

* Skipped entirely (web app default — fast, no GPU needed).
* Instantiated once and reused across many calls (batch experiments).
* Swapped for an API-based backend in the future.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import Any

# ---------------------------------------------------------------------------
# Default model — matches the original monolithic code.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

_GENERATION_ARGS = {
    "max_new_tokens": 2000,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# ---------------------------------------------------------------------------
# Built-in prompt templates.  Use ``{table}`` as the placeholder.
# ---------------------------------------------------------------------------
PROMPT_PAPER_TABLE = (
    "I will provide a table(s) and caption(s) as a list, where "
    "'block_type' identifies the element (e.g., 'table-cell', "
    "'caption') and 'html' contains its content. Your task is to "
    "generate a short, concise, and professional description "
    "(approximately 2-3 sentences) of the table. The description "
    "must synthesize the information from the caption and the "
    "cell data to highlight the main trends, key figures, or most "
    "significant findings presented. Provide the description(s) "
    "as a *valid JSON array* of strings, such as: "
    "['<Table1 description>', '<Table2 description>', ...]. "
    "{table}"
)

PROMPT_MARKDOWN_TABLE = (
    "I will provide a table parsed from a Markdown file. Your task is to "
    "generate a short, concise, and professional description (approximately "
    "2-3 sentences) of the table. The description must synthesize the "
    "information from the table cells to highlight the main trends, key "
    "figures, or most significant findings presented. {table}"
)


class TableSummarizer:
    """Generate short descriptions of tables using a local LLM.

    Parameters
    ----------
    model_path : str
        HuggingFace model identifier (default: ``Qwen/Qwen3-4B-Instruct-2507``).
    device_map : str
        Device mapping strategy (default ``"auto"``).
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        *,
        device_map: str = "auto",
        use_openrouter: bool | None = None,
    ) -> None:
        self.model_path = model_path
        self.device_map = device_map
        self.use_openrouter = use_openrouter
        self._pipe = None  # Lazy-loaded

    def _should_use_openrouter(self) -> bool:
        if self.use_openrouter is not None:
            return self.use_openrouter
        try:
            import torch
            return not (torch.cuda.is_available() or torch.backends.mps.is_available())
        except ImportError:
            return True

    def _call_openrouter(self, messages: list[dict[str, str]]) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required when using OpenRouter.")

        # Use a known valid OpenRouter model for table summarization
        # qwen/qwen-2.5-7b-instruct is fast, capable, and available on OpenRouter
        model = "qwen/qwen-2.5-7b-instruct"

        data = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": _GENERATION_ARGS["temperature"],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8")
            raise RuntimeError(f"OpenRouter API error ({e.code}): {error_msg}")

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _get_pipeline(self):
        """Create the text-generation pipeline on first use."""
        if self._pipe is not None:
            return self._pipe

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return self._pipe

    # ------------------------------------------------------------------
    # Public API — batch (paper JSON tables)
    # ------------------------------------------------------------------
    def summarize(
        self,
        raw_tables: dict[str, list[dict[str, Any]]],
        *,
        prompt: str = PROMPT_PAPER_TABLE,
    ) -> list[str]:
        """Summarise tables extracted by :func:`parse_marker_json`.

        Parameters
        ----------
        raw_tables : dict[str, list[dict]]
            Keyed ``"Table0"``, ``"Table1"``, … — each value is a list of
            marker block dicts with ``block_type`` and ``html`` keys.
        prompt : str
            Prompt template containing ``{table}`` placeholder.

        Returns
        -------
        list[str]
            One natural-language description per table.
        """
        if not raw_tables:
            return []

        # Flatten into a single list of {block_type, html} dicts
        flat: list[dict[str, str]] = []
        for children in raw_tables.values():
            for block in children:
                flat.append(
                    {"block_type": block.get("block_type", ""), "html": block.get("html", "")}
                )

        return self._generate_descriptions(flat, prompt=prompt)

    # ------------------------------------------------------------------
    # Public API — single table (markdown tables, or any arbitrary data)
    # ------------------------------------------------------------------
    def summarize_single(
        self,
        table_data: Any,
        *,
        prompt: str = PROMPT_MARKDOWN_TABLE,
    ) -> str:
        """Summarise a single table and return a plain-text description.

        Parameters
        ----------
        table_data
            Any serialisable representation of a table — typically a
            mistune AST dict for markdown tables, or a list of block dicts
            for paper tables.
        prompt : str
            Prompt template containing ``{table}`` placeholder.

        Returns
        -------
        str
            A short description of the table.
        """
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt.format(table=table_data)},
        ]

        if self._should_use_openrouter():
            return self._call_openrouter(messages)

        pipe = self._get_pipeline()
        output = pipe(messages, **_GENERATION_ARGS)
        return output[0]["generated_text"]

    # ------------------------------------------------------------------
    # Internal: LLM call that returns a list (used by summarize())
    # ------------------------------------------------------------------
    def _generate_descriptions(
        self,
        table_blocks: list[dict[str, str]],
        *,
        prompt: str = PROMPT_PAPER_TABLE,
    ) -> list[str]:
        """Send table blocks to the LLM and parse the response as a JSON array."""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt.format(table=table_blocks)},
        ]

        if self._should_use_openrouter():
            text = self._call_openrouter(messages)
        else:
            pipe = self._get_pipeline()
            output = pipe(messages, **_GENERATION_ARGS)
            text = output[0]["generated_text"]

        # Attempt to parse the response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first JSON-like array from the response
        match = re.search(r"\[(.*?)\]", text, re.S)
        if match:
            items = re.findall(r"'(.*?)'", match.group(1))
            if items:
                return items

        return []
