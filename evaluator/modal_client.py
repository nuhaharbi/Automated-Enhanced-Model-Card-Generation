"""Modal inference client for reproducibility scoring."""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

import requests


def _coerce_eval_payload(value: Any) -> Any:
    """Normalize parsed model payload into expected shape when possible."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        # Some model outputs return only the evaluations list.
        if all(isinstance(x, dict) for x in value):
            return {"evaluations": value}
    return None


def _parse_output_text(text: str) -> Any:
    """Parse model text output with tolerant fallbacks.

    Handles:
    - strict JSON object/list
    - fenced markdown JSON
    - JSON-like substring extraction
    - Python dict/list string via ``ast.literal_eval``
    """
    cleaned = text.strip()

    # Handle markdown fences such as ```json ... ```
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Some models append metadata suffixes.
    if ".assistantfinal" in cleaned:
        cleaned = cleaned.split(".assistantfinal")[-1].strip()

    # 1) Strict JSON parse of full text.
    try:
        return _coerce_eval_payload(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # 2) Extract first object/list candidate and parse as JSON.
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}") + 1
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]") + 1

    candidates: list[str] = []
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(cleaned[obj_start:obj_end])
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(cleaned[arr_start:arr_end])

    for candidate in candidates:
        try:
            return _coerce_eval_payload(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    # 3) Fallback to Python literal parsing for single-quoted dict/list outputs.
    for candidate in [cleaned] + candidates:
        try:
            parsed = ast.literal_eval(candidate)
            return _coerce_eval_payload(parsed)
        except Exception:  # noqa: BLE001
            continue

    return None


def score_prompts_with_modal(
    *,
    endpoint_url: str,
    prompts: list[str],
    system_prompt: str,
    response_schema: dict[str, Any],
    model_name: str = "openai/gpt-oss-20b",
    timeout: int = 180,
    api_key: str | None = None,
) -> list[Any]:
    """Send a batch of prompts to a Modal-hosted scorer endpoint.

    Expected endpoint contract:
    - Request: {
        "model": str,
        "system_prompt": str,
        "prompts": list[str],
        "response_schema": dict
      }
    - Response: {
        "outputs": list[dict|str]
      }
    """
    if not endpoint_url:
        raise ValueError("Modal endpoint URL is required")

    token = api_key or os.getenv("MODAL_EVALUATOR_API_KEY")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "model": model_name,
        "system_prompt": system_prompt,
        "prompts": prompts,
        "response_schema": response_schema,
    }

    resp = requests.post(endpoint_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()
    outputs = data.get("outputs", [])
    if not isinstance(outputs, list):
        raise RuntimeError("Invalid Modal response: 'outputs' must be a list")

    parsed: list[Any] = []
    for item in outputs:
        if isinstance(item, dict):
            parsed.append(item)
            continue
        if isinstance(item, str):
            parsed.append(_parse_output_text(item))
            continue
        parsed.append(None)

    return parsed


def score_prompt_with_modal(
    *,
    endpoint_url: str,
    prompt: str,
    system_prompt: str,
    response_schema: dict[str, Any],
    model_name: str = "openai/gpt-oss-20b",
    timeout: int = 180,
    api_key: str | None = None,
) -> Any:
    """Send a single prompt to a Modal-hosted scorer endpoint."""
    outputs = score_prompts_with_modal(
        endpoint_url=endpoint_url,
        prompts=[prompt],
        system_prompt=system_prompt,
        response_schema=response_schema,
        model_name=model_name,
        timeout=timeout,
        api_key=api_key,
    )
    return outputs[0] if outputs else None
