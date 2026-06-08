"""FastAPI web application for single-model Enhanced Model Card generation.

Run locally:
    uvicorn webapp.app:app --reload
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from contextlib import contextmanager
from queue import Queue
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from classifier.core import classify_preprocessed_paragraphs
from classifier.models import ClassificationResult
from evaluator.core import evaluate_reproducibility
from generator.core import generate_enhanced_model_card
from preprocessor.core import preprocess_markdown, preprocess_paper
from preprocessor.table_summarizer import TableSummarizer
from retriever.core import retrieve_model_artifacts
from retriever.models import RetrievalResult

_STATIC_DIR = Path(__file__).parent / "static"
_TABLE_SUMMARIZER: TableSummarizer | None = None

app = FastAPI(
    title="Enhanced Model Card Generator",
    description="Retrieve artifacts and generate enhanced model cards for HuggingFace models.",
    version="0.3.0",
)

# Mount static assets
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
@contextmanager
def _temporary_env_var(name: str, value: str | None):
    """Temporarily set an environment variable for the current request scope."""
    if not value:
        yield
        return

    old_value = os.getenv(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _get_table_summarizer() -> TableSummarizer:
    global _TABLE_SUMMARIZER
    if _TABLE_SUMMARIZER is None:
        _TABLE_SUMMARIZER = TableSummarizer()
    return _TABLE_SUMMARIZER


def _stream_ndjson_response(
    operation: str,
    worker: Callable[[Callable[[dict[str, Any]], None]], dict[str, Any]],
) -> StreamingResponse:
    event_queue: Queue[Any] = Queue()
    sentinel = object()

    def run_worker() -> None:
        try:
            result = worker(lambda event: event_queue.put({"type": "progress", "operation": operation, **event}))
            event_queue.put({"type": "result", "operation": operation, "data": result})
        except Exception as exc:  # noqa: BLE001
            event_queue.put({"type": "error", "operation": operation, "message": str(exc)})
        finally:
            event_queue.put(sentinel)

    Thread(target=run_worker, daemon=True).start()

    def iterator():
        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(iterator(), media_type="application/x-ndjson")


class RetrieveRequest(BaseModel):
    model_id: str = Field(..., examples=["meta-llama/Llama-2-7b"])


class RetrieveResponse(BaseModel):
    model_id: str
    author: str | None = None
    primary_paper_url: str | None = None
    secondary_paper_urls: list[str] = []
    primary_github_url: str | None = None
    secondary_github_urls: list[str] = []
    tags: list[str] = []
    downloads: int = 0
    likes: int = 0
    # --- Artifact content ---
    model_card: str = ""
    paper_content: str = ""
    github_readme: str = ""
    github_files: list[str] = []
    model_files: list[str] = []
    discussions_count: int = 0


class PreprocessRequest(BaseModel):
    model_id: str = Field(..., examples=["meta-llama/Llama-2-7b"])
    summarize_tables: bool = Field(
        False,
        description=(
            "Generate LLM descriptions for paper tables. "
            "Requires GPU; disabled by default for speed."
        ),
    )


class PreprocessedPaperResponse(BaseModel):
    """Structured preprocessing output for a single paper."""

    sections: dict[str, list[str]] = {}
    sections_split: list[str] = []
    paragraphs_split: list[str] = []
    sentences_split: list[str] = []
    table_descriptions: list[str] = []
    math_map: dict[str, str] = {}
    links_map: dict[str, str] = {}
    success: bool = False


class PreprocessedMarkdownResponse(BaseModel):
    """Structured preprocessing output for a markdown document."""

    sections_split: list[str] = []
    paragraphs_split: list[str] = []
    sentences_split: list[str] = []
    codes_count: int = 0
    tables_count: int = 0
    links: list[str] = []
    emails: list[str] = []
    success: bool = False


class PreprocessResponse(BaseModel):
    model_id: str
    primary_paper_url: str | None = None
    primary_github_url: str | None = None
    paper: PreprocessedPaperResponse = PreprocessedPaperResponse()
    model_card: PreprocessedMarkdownResponse = PreprocessedMarkdownResponse()
    github_readme: PreprocessedMarkdownResponse = PreprocessedMarkdownResponse()


class ClassifyRequest(BaseModel):
    model_id: str = Field(..., examples=["meta-llama/Llama-2-7b"])
    summarize_tables: bool = Field(
        False,
        description="Whether to summarize tables during preprocessing.",
    )
    embedding_backend: Literal["auto", "local", "openrouter"] = Field(
        "auto",
        description="Embedding backend for classifier inference.",
    )
    openrouter_embedding_model: str = Field(
        "qwen/qwen3-embedding-4b",
        description="OpenRouter embedding model ID used when backend is openrouter.",
    )
    use_openrouter_embeddings: bool | None = Field(
        None,
        description="Legacy compatibility flag. When set, overrides embedding_backend=auto.",
    )
    openrouter_api_key: str | None = Field(
        None,
        description="Optional OpenRouter API key override for this request.",
    )


class ParagraphPredictionResponse(BaseModel):
    paragraph: str
    predicted_label: str
    confidence: float
    top2_label: str | None = None
    top2_confidence: float | None = None
    source: str
    index: int
    applied_rerank_rule: bool


class ArtifactClassificationResponse(BaseModel):
    source: str
    predictions: list[ParagraphPredictionResponse] = []


class ClassifyResponse(BaseModel):
    model_id: str
    primary_paper_url: str | None = None
    primary_github_url: str | None = None
    labels: list[str] = []
    paper: ArtifactClassificationResponse = ArtifactClassificationResponse(source="paper")
    model_card: ArtifactClassificationResponse = ArtifactClassificationResponse(source="model_card")
    github_readme: ArtifactClassificationResponse = ArtifactClassificationResponse(source="github_readme")
    tuning_status: dict[str, Any] = {}
    success: bool = False


class EvaluateRequest(BaseModel):
    model_id: str = Field(..., examples=["meta-llama/Llama-2-7b"])
    summarize_tables: bool = Field(
        False,
        description="Whether to summarize tables during preprocessing.",
    )
    embedding_backend: Literal["auto", "local", "openrouter"] = Field(
        "auto",
        description="Embedding backend for classifier inference.",
    )
    openrouter_embedding_model: str = Field(
        "qwen/qwen3-embedding-4b",
        description="OpenRouter embedding model ID used when backend is openrouter.",
    )
    use_openrouter_embeddings: bool | None = Field(
        None,
        description="Legacy compatibility flag. When set, overrides embedding_backend=auto.",
    )
    openrouter_api_key: str | None = Field(
        None,
        description="Optional OpenRouter API key override for this request.",
    )
    modal_model_name: str = Field(
        "openai/gpt-oss-20b",
        description="Modal-hosted LLM model name.",
    )
    modal_endpoint_url: str | None = Field(
        None,
        description="URL of your Modal evaluator endpoint. If empty, uses MODAL_EVALUATOR_URL env var.",
    )


class EvaluationItemResponse(BaseModel):
    point_number: int
    evidence_quote: str
    reasoning: str
    score: float


class SectionEvaluationResponse(BaseModel):
    section: str
    evaluations: list[EvaluationItemResponse] = []


class EvaluateResponse(BaseModel):
    model_id: str
    primary_paper_url: str | None = None
    primary_github_url: str | None = None
    model_name: str
    tuning_status: dict[str, Any] = {}
    scores_by_point: dict[str, float] = {}
    section_scores: dict[str, float] = {}
    total_score: float = 0.0
    section_evaluations: list[SectionEvaluationResponse] = []
    success: bool = False
    error: str | None = None


class GenerateRequest(BaseModel):
    model_id: str = Field(..., examples=["meta-llama/Llama-2-7b"])
    summarize_tables: bool = Field(
        False,
        description="Whether to summarize tables during preprocessing.",
    )
    embedding_backend: Literal["auto", "local", "openrouter"] = Field(
        "auto",
        description="Embedding backend for classifier inference.",
    )
    openrouter_embedding_model: str = Field(
        "qwen/qwen3-embedding-4b",
        description="OpenRouter embedding model ID used when backend is openrouter.",
    )
    use_openrouter_embeddings: bool | None = Field(
        None,
        description="Legacy compatibility flag. When set, overrides embedding_backend=auto.",
    )
    openrouter_api_key: str | None = Field(
        None,
        description="Optional OpenRouter API key override for this request.",
    )
    generator_model_name: str = Field(
        "google/gemini-2.5-flash",
        description="OpenRouter chat model used for section generation.",
    )


class GenerateResponse(BaseModel):
    model_id: str
    primary_paper_url: str | None = None
    primary_github_url: str | None = None
    generator_model_name: str
    sections: dict[str, str] = {}
    full_markdown: str = ""
    placeholder_replacements: dict[str, str] = {}
    tuning_status: dict[str, Any] = {}
    success: bool = False
    error: str | None = None


class SampleGeneratedCardResponse(BaseModel):
    class SampleArtifactChunks(BaseModel):
        source_link: str | None = None
        links: list[str] = []
        paragraphs_split: list[str] = []

    model_id: str
    task: str | None = None
    generator_model_name: str = ""
    repro_score: float | None = None
    repro_report: dict[str, Any] | None = None
    sections: dict[str, str] = {}
    model_card: SampleArtifactChunks = SampleArtifactChunks()
    paper: SampleArtifactChunks = SampleArtifactChunks()
    github_readme: SampleArtifactChunks = SampleArtifactChunks()
    success: bool = True
    error: str | None = None


class SampleCardListItemResponse(BaseModel):
    model_id: str
    task: str | None = None
    repro_score: float | None = None


class SampleCardListResponse(BaseModel):
    total: int
    samples: list[SampleCardListItemResponse] = []


@dataclass
class PipelineContext:
    retrieval: RetrievalResult
    declared_language: str | None
    paper_result: PreprocessedPaperResponse
    model_card_result: PreprocessedMarkdownResponse
    github_result: PreprocessedMarkdownResponse
    paper_obj: Any | None = None
    model_card_obj: Any | None = None
    github_obj: Any | None = None
    paper_paragraphs: list[str] = field(default_factory=list)
    model_card_paragraphs: list[str] = field(default_factory=list)
    github_paragraphs: list[str] = field(default_factory=list)


def _retrieve_model_or_404(model_id: str) -> RetrievalResult:
    retrieval = retrieve_model_artifacts(model_id)

    if not retrieval.model_card:
        raise HTTPException(status_code=404, detail=f"Model card not found for '{model_id}'")

    return retrieval


def _build_pipeline_context(
    retrieval: RetrievalResult,
    *,
    summarize_tables: bool,
) -> PipelineContext:
    declared_language = retrieval.card_data.get("language") if retrieval.card_data else None
    table_summarizer = _get_table_summarizer() if summarize_tables else None

    paper_result = PreprocessedPaperResponse()
    paper_obj: Any | None = None
    paper_paragraphs: list[str] = []
    if retrieval.paper_content:
        pp = preprocess_paper(
            retrieval.paper_content,
            paper_json=retrieval.paper_json,
            declared_language=declared_language,
            summarize_tables=summarize_tables,
            table_summarizer=table_summarizer,
        )
        paper_obj = pp
        paper_result = PreprocessedPaperResponse(
            sections=pp.sections,
            sections_split=pp.sections_split,
            paragraphs_split=pp.paragraphs_split,
            sentences_split=pp.sentences_split,
            table_descriptions=pp.table_descriptions,
            math_map=pp.math_map,
            links_map=pp.links_map,
            success=pp.success,
        )
        paper_paragraphs = pp.paragraphs_split

    model_card_result = PreprocessedMarkdownResponse()
    model_card_obj: Any | None = None
    model_card_paragraphs: list[str] = []
    if retrieval.model_card:
        mc = preprocess_markdown(
            retrieval.model_card,
            declared_language=declared_language,
            summarize_tables=summarize_tables,
            table_summarizer=table_summarizer,
        )
        model_card_obj = mc
        model_card_result = PreprocessedMarkdownResponse(
            sections_split=mc.sections_split,
            paragraphs_split=mc.paragraphs_split,
            sentences_split=mc.sentences_split,
            codes_count=len(mc.codes),
            tables_count=len(mc.tables),
            links=mc.links,
            emails=mc.emails,
            success=mc.success,
        )
        model_card_paragraphs = mc.paragraphs_split

    github_result = PreprocessedMarkdownResponse()
    github_obj: Any | None = None
    github_paragraphs: list[str] = []
    if retrieval.github_readme:
        gh = preprocess_markdown(
            retrieval.github_readme,
            declared_language=declared_language,
            summarize_tables=summarize_tables,
            table_summarizer=table_summarizer,
        )
        github_obj = gh
        github_result = PreprocessedMarkdownResponse(
            sections_split=gh.sections_split,
            paragraphs_split=gh.paragraphs_split,
            sentences_split=gh.sentences_split,
            codes_count=len(gh.codes),
            tables_count=len(gh.tables),
            links=gh.links,
            emails=gh.emails,
            success=gh.success,
        )
        github_paragraphs = gh.paragraphs_split

    return PipelineContext(
        retrieval=retrieval,
        declared_language=declared_language,
        paper_result=paper_result,
        model_card_result=model_card_result,
        github_result=github_result,
        paper_obj=paper_obj,
        model_card_obj=model_card_obj,
        github_obj=github_obj,
        paper_paragraphs=paper_paragraphs,
        model_card_paragraphs=model_card_paragraphs,
        github_paragraphs=github_paragraphs,
    )


def _classify_pipeline_context(
    context: PipelineContext,
    *,
    embedding_backend: Literal["auto", "local", "openrouter"],
    use_openrouter_embeddings: bool | None,
    openrouter_embedding_model: str,
    openrouter_api_key: str | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ClassificationResult:
    use_openrouter = _resolve_openrouter_usage(
        embedding_backend=embedding_backend,
        use_openrouter_embeddings=use_openrouter_embeddings,
    )

    with _temporary_env_var("OPENROUTER_API_KEY", openrouter_api_key):
        return classify_preprocessed_paragraphs(
            context.retrieval.model_id,
            paper_paragraphs=context.paper_paragraphs,
            model_card_paragraphs=context.model_card_paragraphs,
            github_paragraphs=context.github_paragraphs,
            use_openrouter_embeddings=use_openrouter,
            openrouter_embedding_model=openrouter_embedding_model,
            progress_callback=progress_callback,
        )


def _build_evaluate_response(
    req: EvaluateRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> EvaluateResponse:
    retrieval = _retrieve_model_or_404(req.model_id)
    context = _build_pipeline_context(retrieval, summarize_tables=req.summarize_tables)
    cls = _classify_pipeline_context(
        context,
        embedding_backend=req.embedding_backend,
        use_openrouter_embeddings=req.use_openrouter_embeddings,
        openrouter_embedding_model=req.openrouter_embedding_model,
        openrouter_api_key=req.openrouter_api_key,
        progress_callback=progress_callback,
    )

    references = _collect_repro_references(cls)
    files = list(retrieval.github_files) + list(retrieval.model_files)

    modal_endpoint_url = (req.modal_endpoint_url or "").strip() or os.getenv("MODAL_EVALUATOR_URL")

    result = evaluate_reproducibility(
        model_id=retrieval.model_id,
        section_references=references,
        files=files,
        modal_endpoint_url=modal_endpoint_url,
        modal_model_name=req.modal_model_name,
        progress_callback=progress_callback,
    )

    return EvaluateResponse(
        model_id=result.model_id,
        primary_paper_url=retrieval.primary_paper_url,
        primary_github_url=retrieval.primary_github_url,
        model_name=result.model_name,
        tuning_status=cls.tuning_status,
        scores_by_point=result.scores_by_point,
        section_scores=result.section_scores,
        total_score=result.total_score,
        section_evaluations=[
            SectionEvaluationResponse(
                section=sec.section,
                evaluations=[
                    EvaluationItemResponse(
                        point_number=item.point_number,
                        evidence_quote=item.evidence_quote,
                        reasoning=item.reasoning,
                        score=item.score,
                    )
                    for item in sec.evaluations
                ],
            )
            for sec in result.section_evaluations
        ],
        success=result.success,
        error=result.error,
    )


def _build_generate_response(
    req: GenerateRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> GenerateResponse:
    retrieval = _retrieve_model_or_404(req.model_id)
    context = _build_pipeline_context(retrieval, summarize_tables=req.summarize_tables)
    cls = _classify_pipeline_context(
        context,
        embedding_backend=req.embedding_backend,
        use_openrouter_embeddings=req.use_openrouter_embeddings,
        openrouter_embedding_model=req.openrouter_embedding_model,
        openrouter_api_key=req.openrouter_api_key,
        progress_callback=progress_callback,
    )

    placeholder_replacements, source_placeholder_maps = _build_placeholder_replacements(
        paper=context.paper_obj,
        model_card=context.model_card_obj,
        github=context.github_obj,
    )

    gen_refs = _collect_generation_references(
        cls,
        source_placeholder_maps=source_placeholder_maps,
    )
    with _temporary_env_var("OPENROUTER_API_KEY", req.openrouter_api_key):
        generated = generate_enhanced_model_card(
            model_id=retrieval.model_id,
            section_references=gen_refs,
            model_name=req.generator_model_name,
            progress_callback=progress_callback,
        )

    resolved_sections = {
        name: _apply_placeholder_replacements(content, placeholder_replacements)
        for name, content in generated.sections.items()
    }
    resolved_full_markdown = _apply_placeholder_replacements(
        generated.full_markdown,
        placeholder_replacements,
    )

    return GenerateResponse(
        model_id=generated.model_id,
        primary_paper_url=retrieval.primary_paper_url,
        primary_github_url=retrieval.primary_github_url,
        generator_model_name=generated.model_name,
        sections=resolved_sections,
        full_markdown=resolved_full_markdown,
        placeholder_replacements=placeholder_replacements,
        tuning_status=cls.tuning_status,
        success=generated.success,
        error=generated.error,
    )


def _to_classify_response(
    result: ClassificationResult,
    *,
    primary_paper_url: str | None,
    primary_github_url: str | None,
) -> ClassifyResponse:
    return ClassifyResponse(
        model_id=result.model_id,
        primary_paper_url=primary_paper_url,
        primary_github_url=primary_github_url,
        labels=result.labels,
        paper=ArtifactClassificationResponse(
            source=result.paper.source,
            predictions=[ParagraphPredictionResponse(**p.__dict__) for p in result.paper.predictions],
        ),
        model_card=ArtifactClassificationResponse(
            source=result.model_card.source,
            predictions=[ParagraphPredictionResponse(**p.__dict__) for p in result.model_card.predictions],
        ),
        github_readme=ArtifactClassificationResponse(
            source=result.github_readme.source,
            predictions=[ParagraphPredictionResponse(**p.__dict__) for p in result.github_readme.predictions],
        ),
        tuning_status=result.tuning_status,
        success=result.success,
    )


def _collect_repro_references(cls: ClassificationResult) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}

    all_preds = (
        cls.paper.predictions
        + cls.model_card.predictions
        + cls.github_readme.predictions
    )

    for pred in all_preds:
        label = pred.predicted_label
        refs.setdefault(label, []).append(pred.paragraph)

    return refs


def _resolve_openrouter_usage(
    *,
    embedding_backend: Literal["auto", "local", "openrouter"],
    use_openrouter_embeddings: bool | None,
) -> bool:
    if embedding_backend == "local":
        return False
    if embedding_backend == "openrouter":
        return True
    if use_openrouter_embeddings is not None:
        return use_openrouter_embeddings
    return True


from .sample_cards import (
    _SAMPLE_GENERATED_CARDS_PATH,
    _build_csv_backed_artifact_chunks_for_model,
    _build_csv_backed_replacements_for_model,
    _build_csv_backed_source_payloads_for_model,
    _build_placeholder_replacements,
    _apply_placeholder_replacements,
    _collect_generation_references,
    _load_repro_report_cache,
    _load_repro_score_cache,
    _normalize_formula_text,
    _normalize_sample_placeholder_prefixes,
    _resolve_remaining_placeholders_for_section,
    _strip_unresolved_code_placeholders,
    _upgrade_legacy_placeholders_with_citations,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the main web UI."""
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/sample-generated-card", response_model=SampleGeneratedCardResponse)
def sample_generated_card(model_id: str | None = None) -> SampleGeneratedCardResponse:
    """Return one generated card sample from task-stratified JSONL output."""
    if not _SAMPLE_GENERATED_CARDS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Sample file not found at "
                f"'{_SAMPLE_GENERATED_CARDS_PATH}'. "
                "Run full pipeline eval first or update the path."
            ),
        )

    selected: dict[str, Any] | None = None
    with _SAMPLE_GENERATED_CARDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_model_id = str(row.get("modelId", "")).strip()
            if not selected:
                selected = row

            if model_id and row_model_id == model_id.strip():
                selected = row
                break

    if not selected:
        raise HTTPException(
            status_code=404,
            detail="No sample generated cards were found in the JSONL file.",
        )

    generated_card = selected.get("generated_card")
    sections = generated_card if isinstance(generated_card, dict) else {}

    if model_id and str(selected.get("modelId", "")).strip() != model_id.strip():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' was not found in the sample generated cards file.",
        )

    model_id_value = str(selected.get("modelId", ""))
    csv_replacements = _build_csv_backed_replacements_for_model(model_id_value)
    source_payloads = _build_csv_backed_source_payloads_for_model(model_id_value)
    artifact_chunks = _build_csv_backed_artifact_chunks_for_model(model_id_value)
    repro_score = _load_repro_score_cache().get(model_id_value)
    repro_report = _load_repro_report_cache().get(model_id_value)
    if repro_report is not None and "success" not in repro_report:
        repro_report = {**repro_report, "success": True}

    resolved_sections: dict[str, str] = {}
    for k, v in sections.items():
        section_name = str(k)
        upgraded = _upgrade_legacy_placeholders_with_citations(str(v))
        normalized = _normalize_sample_placeholder_prefixes(upgraded)
        replaced = _apply_placeholder_replacements(normalized, csv_replacements)
        resolved = _resolve_remaining_placeholders_for_section(
            replaced,
            section_name=section_name,
            source_payloads=source_payloads,
        )
        resolved = _strip_unresolved_code_placeholders(resolved)
        resolved_sections[section_name] = _normalize_formula_text(resolved)

    return SampleGeneratedCardResponse(
        model_id=model_id_value,
        task=selected.get("task"),
        generator_model_name=str(selected.get("generator_model", "")),
        repro_score=repro_score,
        repro_report=repro_report,
        sections=resolved_sections,
        model_card=SampleGeneratedCardResponse.SampleArtifactChunks(**artifact_chunks.get("model_card", {})),
        paper=SampleGeneratedCardResponse.SampleArtifactChunks(**artifact_chunks.get("paper", {})),
        github_readme=SampleGeneratedCardResponse.SampleArtifactChunks(**artifact_chunks.get("github_readme", {})),
        success=True,
        error=str(selected.get("error", "")) or None,
    )


@app.get("/sample-generated-cards", response_model=SampleCardListResponse)
def sample_generated_cards() -> SampleCardListResponse:
    """Return all sample model IDs available in the task-stratified generated-card artifact."""
    if not _SAMPLE_GENERATED_CARDS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Sample file not found at "
                f"'{_SAMPLE_GENERATED_CARDS_PATH}'. "
                "Run full pipeline eval first or update the path."
            ),
        )

    repro_scores = _load_repro_score_cache()
    samples: list[SampleCardListItemResponse] = []
    seen: set[str] = set()

    with _SAMPLE_GENERATED_CARDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue

            model_id = str(row.get("modelId", "")).strip()
            if not model_id or model_id in seen:
                continue

            seen.add(model_id)
            samples.append(
                SampleCardListItemResponse(
                    model_id=model_id,
                    task=row.get("task"),
                    repro_score=repro_scores.get(model_id),
                )
            )

    samples.sort(key=lambda x: x.model_id.lower())
    return SampleCardListResponse(total=len(samples), samples=samples)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Retrieve all artifacts for a HuggingFace model ID.

    This is the first stage of the Enhanced Model Card pipeline.
    Later stages (classification, LLM generation) will be added as
    additional endpoints.
    """
    result = _retrieve_model_or_404(req.model_id)

    return RetrieveResponse(
        model_id=result.model_id,
        author=result.author,
        primary_paper_url=result.primary_paper_url,
        secondary_paper_urls=result.secondary_paper_urls,
        primary_github_url=result.primary_github_url,
        secondary_github_urls=result.secondary_github_urls,
        tags=result.tags,
        downloads=result.downloads,
        likes=result.likes,
        model_card=result.model_card,
        paper_content=result.paper_content,
        github_readme=result.github_readme,
        github_files=result.github_files,
        model_files=result.model_files,
        discussions_count=len(result.model_discussions),
    )


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest) -> PreprocessResponse:
    """Retrieve artifacts and preprocess paper, model card, and README.

    Runs the full retrieval pipeline (Component 1) followed by
    preprocessing (Component 2) on all three artifact types.
    """
    retrieval = _retrieve_model_or_404(req.model_id)
    context = _build_pipeline_context(retrieval, summarize_tables=req.summarize_tables)

    return PreprocessResponse(
        model_id=retrieval.model_id,
        primary_paper_url=retrieval.primary_paper_url,
        primary_github_url=retrieval.primary_github_url,
        paper=context.paper_result,
        model_card=context.model_card_result,
        github_readme=context.github_result,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """Retrieve, preprocess, and classify paragraph splits (Component 3)."""
    retrieval = _retrieve_model_or_404(req.model_id)
    context = _build_pipeline_context(retrieval, summarize_tables=req.summarize_tables)
    cls = _classify_pipeline_context(
        context,
        embedding_backend=req.embedding_backend,
        use_openrouter_embeddings=req.use_openrouter_embeddings,
        openrouter_embedding_model=req.openrouter_embedding_model,
        openrouter_api_key=req.openrouter_api_key,
    )

    return _to_classify_response(
        cls,
        primary_paper_url=retrieval.primary_paper_url,
        primary_github_url=retrieval.primary_github_url,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Retrieve, preprocess, classify, and score reproducibility (Component 4)."""
    return _build_evaluate_response(req)


@app.post("/evaluate-progress")
def evaluate_progress(req: EvaluateRequest) -> StreamingResponse:
    """Stream classification and scoring progress for the reproducibility endpoint."""

    def worker(progress_callback: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        return _build_evaluate_response(req, progress_callback=progress_callback).model_dump()

    return _stream_ndjson_response("evaluate", worker)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Run full pipeline and generate an Enhanced Model Card (Component 5)."""
    return _build_generate_response(req)


@app.post("/generate-progress")
def generate_progress(req: GenerateRequest) -> StreamingResponse:
    """Stream classification and generation progress for the model-card endpoint."""

    def worker(progress_callback: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        return _build_generate_response(req, progress_callback=progress_callback).model_dump()

    return _stream_ndjson_response("generate", worker)
