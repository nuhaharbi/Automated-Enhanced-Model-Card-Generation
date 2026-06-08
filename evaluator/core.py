"""Core orchestration for reproducibility scoring (Component 4)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from evaluator.modal_client import score_prompt_with_modal, score_prompts_with_modal
from evaluator.models import EvaluationItem, ReproducibilityResult, SectionEvaluation

_RESOURCE_PATH = Path(__file__).parent / "resources" / "reproducibility_checklist.json"


def _load_resource() -> dict[str, Any]:
    return json.loads(_RESOURCE_PATH.read_text(encoding="utf-8"))


def _extract_point_number(checklist_text: str) -> int | None:
    match = re.search(r"\[Point number\s+(\d+)\]", checklist_text)
    if not match:
        return None
    return int(match.group(1))


def _build_user_prompt(reference_chunks: list[str], checklist_items: list[str]) -> str:
    formatted_checklist = "\n".join(checklist_items)
    text = str(reference_chunks)

    return f"""
<checklist>
{formatted_checklist}
</checklist>

<reference_text>
{text}
</reference_text>

<instructions>
For each item in the <checklist>:
1. Identify the requirement.
2. Search <reference_text> for evidence.
3. Extract a direct quote if found.
4. Detailed reasoning for your score based on the evaluation rubric.
5. Output the result in the JSON format specified in the output schema.
</instructions>

<constraints>
1. Source: Base the answer ONLY on the reference_text.
2. Completeness: You MUST evaluate ALL points in the checklist.
3. Mapping: Maintain the exact order and Point Numbers provided in the checklist.
4. Handling File Names: If a file name (e.g., requirements.txt, data.csv) appears in reference_text and standard conventions suggest it contains the required info, treat this as Score 1 (Pass).
5. Handling Placeholders: If placeholders like [1:CODE], [2:MATH], or [3:LINK] appear and surrounding text confirms they represent the required item, treat this as Score 1 (Pass).
</constraints>
""".strip()


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point_number": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 22,
                        },
                        "evidence_quote": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "score": {"type": "number", "enum": [0.0, 0.5, 1.0]},
                    },
                    "required": ["point_number", "evidence_quote", "reasoning", "score"],
                },
            }
        },
        "required": ["evaluations"],
    }


def _coerce_score(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v not in (0.0, 0.5, 1.0):
        return 0.0
    return v


def evaluate_reproducibility(
    *,
    model_id: str,
    section_references: dict[str, list[str]],
    files: list[str] | None = None,
    modal_endpoint_url: str | None = None,
    modal_model_name: str = "openai/gpt-oss-20b",
    modal_api_key: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ReproducibilityResult:
    """Evaluate reproducibility checklist using Modal-hosted LLM inference."""
    resource = _load_resource()
    system_prompt = resource["system_prompt"]
    checklist: dict[str, list[str]] = resource["checklist"]

    endpoint_url = modal_endpoint_url or os.getenv("MODAL_EVALUATOR_URL", "")
    if not endpoint_url:
        return ReproducibilityResult(
            model_id=model_id,
            model_name=modal_model_name,
            success=False,
            error="MODAL_EVALUATOR_URL is not configured",
        )

    sections = list(checklist.keys())
    prompts: list[str] = []

    for section in sections:
        refs = list(section_references.get(section, []))
        if section == "Files" and files:
            refs.extend(files)
        prompts.append(_build_user_prompt(refs, checklist[section]))

    outputs: list[Any] = []
    if progress_callback is None:
        try:
            outputs = score_prompts_with_modal(
                endpoint_url=endpoint_url,
                prompts=prompts,
                system_prompt=system_prompt,
                response_schema=_response_schema(),
                model_name=modal_model_name,
                api_key=modal_api_key,
            )
        except Exception as exc:  # noqa: BLE001
            return ReproducibilityResult(
                model_id=model_id,
                model_name=modal_model_name,
                success=False,
                error=str(exc),
            )
    else:
        total_sections = len(sections)

        def emit(event: dict[str, Any]) -> None:
            progress_callback(event)

        emit(
            {
                "phase": "scoring",
                "stage": "start",
                "completed": 0,
                "total": total_sections,
                "remaining": total_sections,
                "message": (
                    f"Scoring {total_sections} checklist sections."
                    if total_sections
                    else "No checklist sections are available for scoring."
                ),
            }
        )

        for index, section in enumerate(sections, start=1):
            remaining = max(total_sections - index, 0)
            emit(
                {
                    "phase": "scoring",
                    "section": section,
                    "section_index": index,
                    "section_total": total_sections,
                    "completed": index - 1,
                    "total": total_sections,
                    "remaining": remaining + 1,
                    "message": (
                        f"Scoring section {index}/{total_sections}: {section} ({remaining + 1} remaining)."
                        if total_sections
                        else f"No checklist sections are available for {section}."
                    ),
                }
            )

            try:
                out = score_prompt_with_modal(
                    endpoint_url=endpoint_url,
                    prompt=prompts[index - 1],
                    system_prompt=system_prompt,
                    response_schema=_response_schema(),
                    model_name=modal_model_name,
                    api_key=modal_api_key,
                )
            except Exception as exc:  # noqa: BLE001
                emit(
                    {
                        "phase": "scoring",
                        "stage": "error",
                        "section": section,
                        "section_index": index,
                        "section_total": total_sections,
                        "completed": index - 1,
                        "total": total_sections,
                        "remaining": remaining + 1,
                        "message": f"Scoring failed at section {index}/{total_sections}: {section} ({str(exc)[:300]}).",
                    }
                )
                return ReproducibilityResult(
                    model_id=model_id,
                    model_name=modal_model_name,
                    success=False,
                    error=str(exc),
                )

            outputs.append(out)

            emit(
                {
                    "phase": "scoring",
                    "section": section,
                    "section_index": index,
                    "section_total": total_sections,
                    "completed": index,
                    "total": total_sections,
                    "remaining": remaining,
                    "message": (
                        f"Finished section {index}/{total_sections}: {section} ({remaining} remaining)."
                        if total_sections
                        else f"No checklist sections are available for {section}."
                    ),
                }
            )

        emit(
            {
                "phase": "scoring",
                "stage": "complete",
                "completed": total_sections,
                "total": total_sections,
                "remaining": 0,
                "message": "Scoring complete.",
            }
        )

    scores_by_point: dict[str, float] = {}
    section_scores: dict[str, float] = {}
    section_evaluations: list[SectionEvaluation] = []

    for idx, section in enumerate(sections):
        out = outputs[idx] if idx < len(outputs) else None
        eval_items_raw = out.get("evaluations", []) if isinstance(out, dict) else []

        eval_items: list[EvaluationItem] = []
        for raw in eval_items_raw:
            if not isinstance(raw, dict):
                continue
            point_number = int(raw.get("point_number", 0))
            if point_number <= 0:
                continue

            evidence_quote = str(raw.get("evidence_quote", "None"))
            reasoning = str(raw.get("reasoning", ""))
            score = _coerce_score(raw.get("score", 0.0))

            eval_items.append(
                EvaluationItem(
                    point_number=point_number,
                    evidence_quote=evidence_quote,
                    reasoning=reasoning,
                    score=score,
                )
            )
            scores_by_point[str(point_number)] = score

        if not eval_items:
            for checklist_item in checklist[section]:
                pnum = _extract_point_number(checklist_item)
                if pnum is None:
                    continue
                scores_by_point[str(pnum)] = 0.0
                eval_items.append(
                    EvaluationItem(
                        point_number=pnum,
                        evidence_quote="None",
                        reasoning="No structured output returned.",
                        score=0.0,
                    )
                )

        section_total = round(sum(item.score for item in eval_items), 3)
        section_scores[section] = section_total
        section_evaluations.append(SectionEvaluation(section=section, evaluations=eval_items))

    total_score = round(sum(scores_by_point.values()), 3)

    return ReproducibilityResult(
        model_id=model_id,
        model_name=modal_model_name,
        scores_by_point=scores_by_point,
        section_scores=section_scores,
        total_score=total_score,
        section_evaluations=section_evaluations,
        success=True,
    )
