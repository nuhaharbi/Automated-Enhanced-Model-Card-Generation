"""Core orchestration for Enhanced Model Card generation (Component 5)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from generator.models import EnhancedModelCardResult

_RESOURCE_PATH = Path(__file__).parent / "resources" / "section_specs.json"


def _load_resource() -> dict[str, Any]:
    return json.loads(_RESOURCE_PATH.read_text(encoding="utf-8"))


def _build_user_prompt(reference_text: list[str], section_spec: str, section_name: str) -> str:
    formatted_references = "\n".join(f"- {item}" for item in reference_text)
    return f"""
<task>
    Synthesize the provided <reference_text> into a highly readable "{section_name}" section for a Model Card.
    Adhere strictly to the <section_specifications> and <formatting_rules>.
</task>

<section_specifications>
{section_spec}
</section_specifications>

<reference_text>
{formatted_references}
</reference_text>

<instructions>
    1. Synthesize: Combine info from all sources. If sources disagree, state BOTH values clearly.
    2. Cite: Append a citation tag (e.g., [RP]) to the end of every sentence or bullet point derived from source material.
    3. Placeholders: Preserve tokens like [1:CODE], [2:MATH], [3:EMAIL], or [4:LINK] exactly as they appear.
    4. Missing info: If a specification cannot be met with the provided text, strictly state "No info available".
    5. Formatting: You MUST use markdown lists, bolding for key terms, and avoid blocks of text.
</instructions>

<output_format>
### {section_name}
- **[Key Term/Concept]:** [Synthesized detail with citation tag]
- **[Key Term/Concept]:** [Synthesized detail with citation tag]
- [Additional scannable details or short sentences...]
</output_format>
""".strip()


def _chat_completion_openrouter(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    timeout_seconds: int = 120,
) -> str:
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8")
        raise RuntimeError(f"OpenRouter chat HTTP {exc.code}: {msg}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter chat returned no choices: {str(body)[:300]}")

    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"OpenRouter chat returned empty message content: {str(body)[:300]}")
    return content.strip()


def _empty_section(section_name: str) -> str:
    return f"### {section_name}\n\n[More information needed]"


def generate_enhanced_model_card(
    *,
    model_id: str,
    section_references: dict[str, list[str]],
    model_name: str = "google/gemini-2.5-flash",
    openrouter_api_key: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> EnhancedModelCardResult:
    """Generate all Enhanced Model Card sections from classified references."""
    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return EnhancedModelCardResult(
            model_id=model_id,
            model_name=model_name,
            success=False,
            error="OPENROUTER_API_KEY is not configured",
        )

    resource = _load_resource()
    system_prompt = resource["system_prompt"]
    section_specs: dict[str, str] = resource["section_specs"]
    section_names = list(section_specs.keys())
    total_sections = len(section_names)

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "generation",
                "stage": "start",
                "completed": 0,
                "total": total_sections,
                "remaining": total_sections,
                "message": (
                    f"Generating {total_sections} model card sections."
                    if total_sections
                    else "No model card sections are available for generation."
                ),
            }
        )

    generated: dict[str, str] = {}

    for index, (section_name, section_spec) in enumerate(section_specs.items(), start=1):
        refs = list(section_references.get(section_name, []))
        section_remaining = max(total_sections - index, 0)

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "generation",
                    "section": section_name,
                    "section_index": index,
                    "section_total": total_sections,
                    "completed": index - 1,
                    "total": total_sections,
                    "remaining": section_remaining + 1,
                    "message": (
                        f"Generating section {index}/{total_sections}: {section_name} ({section_remaining + 1} remaining)."
                        if total_sections
                        else f"No model card sections are available for {section_name}."
                    ),
                }
            )

        if not refs:
            generated[section_name] = _empty_section(section_name)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "generation",
                        "section": section_name,
                        "section_index": index,
                        "section_total": total_sections,
                        "completed": index,
                        "total": total_sections,
                        "remaining": section_remaining,
                        "message": (
                            f"Finished section {index}/{total_sections}: {section_name} ({section_remaining} remaining)."
                            if total_sections
                            else f"No model card sections are available for {section_name}."
                        ),
                    }
                )
            continue

        user_prompt = _build_user_prompt(refs, section_spec, section_name)
        try:
            output = _chat_completion_openrouter(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=api_key,
            )
            generated[section_name] = output
        except Exception as exc:  # noqa: BLE001
            generated[section_name] = (
                f"### {section_name}\n\n"
                f"[Generation Failed: {str(exc)[:500]}]"
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "generation",
                    "section": section_name,
                    "section_index": index,
                    "section_total": total_sections,
                    "completed": index,
                    "total": total_sections,
                    "remaining": section_remaining,
                    "message": (
                        f"Finished section {index}/{total_sections}: {section_name} ({section_remaining} remaining)."
                        if total_sections
                        else f"No model card sections are available for {section_name}."
                    ),
                }
            )

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "generation",
                "stage": "complete",
                "completed": total_sections,
                "total": total_sections,
                "remaining": 0,
                "message": "Generation complete.",
            }
        )

    full_markdown = "\n\n".join(generated[name] for name in section_specs.keys())

    return EnhancedModelCardResult(
        model_id=model_id,
        model_name=model_name,
        sections=generated,
        full_markdown=full_markdown,
        success=True,
    )
