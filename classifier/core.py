"""Paragraph-only classifier entry points.

Primary runtime path:
1. Qwen paragraph embeddings + label centroid similarity.
2. Rule 1: rerank when top-1/top-2 gap <= configured threshold.
3. Rule 2: rerank for configured beneficial top-1/top-2 pairs.

Fallback path:
- Keyword overlap scoring when sentence-transformers is unavailable.
"""

from __future__ import annotations

import json
import importlib
import os
import re
import urllib.error
import urllib.request
from csv import DictReader
from pathlib import Path
from typing import Any, Callable

import numpy as np

from classifier.models import ArtifactClassification, ClassificationResult, ParagraphPrediction

_SCHEMA_PATH = Path(__file__).parent / "resources" / "paragraph_schema.json"

_EMBED_MODEL = None
_CENTROIDS_CACHE: tuple[list[str], np.ndarray] | None = None
_OPENROUTER_CENTROIDS_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}
_RULES_CACHE: dict[str, tuple[float, set[tuple[str, str]], dict[str, Any]]] = {}
_DEFAULT_CALIBRATION_FILES: list[tuple[str, str]] = [
    ("labeled_paragraphs.csv", "paragraphs"),
    ("labeled_sentences.csv", "sentences"),
    ("labeled_sections.csv", "sections"),
]

_LABEL_KEYWORDS: dict[str, list[str]] = {
    "Model details": [
        "base model",
        "model architecture",
        "model backbone",
        "based on transformer",
        "encoder-decoder",
        "developed by",
        "release date",
        "license type",
        "reference paper",
        "disclaimer",
        "number of layers",
        "neural network",
        "model size",
        "technical report",
        "cite",
        "recurrent neural networks",
        "attention layer",
        "vision transformer",
        "transformer-based model",
        "model parameters count",
        "model components",
        "model design",
        "model variants",
        "underlying structure",
        "supported tasks",
        "model input",
        "model output",
        "acknowledgement",
        "thanks to",
        "funded by",
        "expected use",
        "main use",
    ],
    "Limitations, bias, and risk": [
        "biased predictions",
        "fairness",
        "ethical concern",
        "risk",
        "sensitive content",
        "stereotype",
        "caution",
        "unfiltered content",
        "content warning",
        "representational harms",
        "malicious content",
        "lead to limitations",
        "influence model capabilities",
        "prejudiced content",
        "inherited bias",
        "demeaning portrayals",
        "ethical implications",
        "inductive bias",
        "bias",
        "hallucination",
        "mitigation strategies",
        "out-of-scope",
        "unsuitable",
    ],
    "Training": [
        "data collection",
        "training dataset",
        "training procedures",
        "epoch count",
        "optimizer type",
        "trained on gpu",
        "trained on tpu",
        "batch size",
        "learning rate",
        "warmup rate",
        "hyperparameters details",
        "loss function",
        "data augmentation",
        "preprocessing",
        "fine-tuning procedures",
        "post-training",
        "pre-training",
        "training approach",
        "data filtering",
        "data preparation process",
        "how to train",
        "training",
        "fine-tuning",
        "optimizer",
        "hyperparameters",
    ],
    "How to use": [
        "quick start",
        "code snippet",
        "setup instructions",
        "reproducibility",
        "installation guide",
        "adjustable configuration",
        "usage tutorial",
        "install dependencies",
        "usage documentation",
        "usage requirements",
        "how to start",
        "online demo",
        "showcase samples",
        "output examples",
        "task demonstration",
        "web demo",
        "inference script",
        "run following code",
        "config file",
        "installation",
        "usage",
        "inference",
        "run",
        "demo",
    ],
    "Evaluation": [
        "model achieve",
        "evaluation result",
        "results on benchmark",
        "automatic evaluation metrics",
        "test set",
        "quantitative analysis",
        "qualitative analysis",
        "error rate",
        "outperform",
        "superior performance",
        "comparable performance",
        "evaluation score",
        "evaluation setup",
        "held-out data",
        "performance comparison",
        "state-of-the-art results",
        "experiment",
        "benchmark",
        "evaluation",
        "results",
        "state of the art",
        "metric",
    ],
    "Environmental impact": [
        "carbon emission",
        "cloud provider",
        "energy consumption",
        "compute cost",
        "power usage",
        "region",
        "total emissions",
        "average carbon efficiency",
        "carbon footprint",
        "cumulative hours",
        "cumulative days",
        "electricity",
        "power grid",
        "carbon",
        "emission",
        "energy",
        "power",
    ],
    "Contact": [
        "contact",
        "corresponding author",
        "send feedback",
        "contributors",
        "affiliation",
        "submit github issue",
        "send questions",
        "send comments",
        "send email",
        "for help",
        "email",
        "issue",
        "feedback",
        "author",
    ],
}


def _load_schema() -> dict:
    data = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return data


def _load_labels() -> list[str]:
    data = _load_schema()
    return list(data.get("labels", []))


def _load_threshold() -> float:
    data = _load_schema()
    return float(data.get("optimal_gap_threshold", 0.0))


def _load_beneficial_pairs() -> set[tuple[str, str]]:
    data = _load_schema()
    raw_pairs = data.get("beneficial_pairs", [])
    pairs: set[tuple[str, str]] = set()
    for pair in raw_pairs:
        if isinstance(pair, list) and len(pair) == 2:
            pairs.add((str(pair[0]), str(pair[1])))
    return pairs


def _load_model_name() -> str:
    data = _load_schema()
    return str(data.get("model_name", "Qwen/Qwen3-Embedding-4B"))


def _load_openrouter_model_name() -> str:
    data = _load_schema()
    return str(data.get("openrouter_model_name", "qwen/qwen3-embedding-4b"))


def _resolve_openrouter_model_name(openrouter_model_name: str | None) -> str:
    if openrouter_model_name and openrouter_model_name.strip():
        return openrouter_model_name.strip()

    env_name = os.getenv("CLASSIFIER_OPENROUTER_EMBEDDING_MODEL", "").strip()
    if env_name:
        return env_name

    return _load_openrouter_model_name()


def _rules_cache_key(*, use_openrouter: bool, openrouter_model_name: str | None) -> str:
    if use_openrouter:
        return f"openrouter:{_resolve_openrouter_model_name(openrouter_model_name)}"
    return f"local:{_load_model_name()}"


def _default_rules() -> tuple[float, set[tuple[str, str]]]:
    return (_load_threshold(), _load_beneficial_pairs())


def _predict_with_rules(
    texts: list[str],
    labels: list[str],
    top1_idx: np.ndarray,
    top2_idx: np.ndarray,
    top1_scores: np.ndarray,
    top2_scores: np.ndarray,
    *,
    threshold: float,
    beneficial_pairs: set[tuple[str, str]],
) -> list[str]:
    preds: list[str] = []
    for i in range(len(texts)):
        p1 = int(top1_idx[i])
        p2 = int(top2_idx[i])
        gap = float(top1_scores[i] - top2_scores[i])
        pair = (labels[p1], labels[p2])

        final_idx = p1
        if gap <= threshold or pair in beneficial_pairs:
            final_idx = _keyword_rerank_index(texts[i], p1, p2, labels)
        preds.append(labels[final_idx])
    return preds


def _stratified_split(
    texts: list[str],
    y_true: list[str],
    *,
    test_size: float = 0.6,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    by_label: dict[str, list[int]] = {}
    for i, y in enumerate(y_true):
        by_label.setdefault(y, []).append(i)

    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _, idxs in by_label.items():
        local = list(idxs)
        rng.shuffle(local)
        n_test = int(round(len(local) * test_size))
        n_test = max(1, min(len(local) - 1, n_test)) if len(local) > 1 else len(local)
        test_idx.extend(local[:n_test])
        val_idx.extend(local[n_test:])

    if not val_idx:
        val_idx = test_idx[: max(1, len(test_idx) // 2)]
    if not test_idx:
        test_idx = val_idx[:]

    val_text = [texts[i] for i in val_idx]
    test_text = [texts[i] for i in test_idx]
    val_y = [y_true[i] for i in val_idx]
    test_y = [y_true[i] for i in test_idx]
    return val_text, test_text, val_y, test_y


def _top2_from_similarity(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(sim, axis=1)[:, ::-1]
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if order.shape[1] > 1 else order[:, 0]
    top1_scores = sim[np.arange(sim.shape[0]), top1_idx]
    top2_scores = sim[np.arange(sim.shape[0]), top2_idx]
    return top1_idx, top2_idx, top1_scores, top2_scores


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    if not labels:
        return 0.0

    f1_scores: list[float] = []
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for t, p in zip(y_true, y_pred):
            if p == label and t == label:
                tp += 1
            elif p == label and t != label:
                fp += 1
            elif p != label and t == label:
                fn += 1

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))

    return float(sum(f1_scores) / len(f1_scores))


def _load_calibration_rows() -> tuple[list[str], list[str]] | None:
    bundle = _load_calibration_bundle()
    if bundle is None:
        return None
    return bundle["texts"], bundle["labels"]


def _load_calibration_bundle() -> dict[str, Any] | None:
    csv_path = os.getenv("CLASSIFIER_CALIBRATION_CSV", "").strip()
    label_col = os.getenv("CLASSIFIER_CALIBRATION_LABEL_COLUMN", "gt").strip() or "gt"
    limit_raw = os.getenv("CLASSIFIER_CALIBRATION_MAX_ROWS", "0").strip()
    max_rows = int(limit_raw) if limit_raw.isdigit() else 0

    sources: list[tuple[Path, str]] = []

    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            return None
        text_col = os.getenv("CLASSIFIER_CALIBRATION_TEXT_COLUMN", "paragraphs").strip() or "paragraphs"
        sources.append((path, text_col))
    else:
        workspace_root = Path(__file__).resolve().parents[1]
        roots = [
            Path.cwd(),
            workspace_root,
            workspace_root / "Datasets",
            Path.cwd() / "Datasets",
        ]
        seen_paths: set[Path] = set()
        for root in roots:
            for file_name, text_col in _DEFAULT_CALIBRATION_FILES:
                p = (root / file_name).resolve()
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                if p.exists():
                    sources.append((p, text_col))

    if not sources:
        return None

    texts: list[str] = []
    labels: list[str] = []
    used_sources: list[dict[str, str]] = []

    for path, text_col in sources:
        loaded_from_source = 0
        with open(path, encoding="utf-8") as f:
            reader = DictReader(f)
            for row in reader:
                text = str(row.get(text_col, "") or "").strip()
                label = str(row.get(label_col, "") or "").strip()
                if not text or not label:
                    continue
                texts.append(text)
                labels.append(label)
                loaded_from_source += 1
                if max_rows > 0 and len(texts) >= max_rows:
                    break
        if loaded_from_source > 0:
            used_sources.append({"path": str(path), "text_column": text_col})
        if max_rows > 0 and len(texts) >= max_rows:
            break

    if not texts:
        return None
    return {
        "texts": texts,
        "labels": labels,
        "sources": used_sources,
        "label_column": label_col,
        "max_rows": max_rows,
    }


def _compute_doc_embeddings(
    texts: list[str],
    *,
    use_openrouter: bool,
    openrouter_model_name: str | None,
) -> np.ndarray | None:
    if use_openrouter:
        emb = _openrouter_embed_texts(texts, openrouter_model_name=openrouter_model_name)
    else:
        model = _maybe_get_embed_model()
        if model is None:
            return None
        emb = model.encode(texts, convert_to_numpy=True)

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def _tune_rules_for_model(
    labels: list[str],
    *,
    use_openrouter: bool,
    openrouter_model_name: str | None,
) -> tuple[float, set[tuple[str, str]], dict[str, Any]] | None:
    bundle = _load_calibration_bundle()
    if bundle is None:
        return None

    texts = bundle["texts"]
    y_true = bundle["labels"]
    if len(texts) < 20:
        return None

    val_texts, test_texts, y_val, y_test = _stratified_split(
        texts,
        y_true,
        test_size=0.6,
        seed=42,
    )

    centroids = _compute_label_centroids(
        labels,
        use_openrouter=use_openrouter,
        openrouter_model_name=openrouter_model_name,
    )
    if centroids is None:
        return None

    docs_val = _compute_doc_embeddings(
        val_texts,
        use_openrouter=use_openrouter,
        openrouter_model_name=openrouter_model_name,
    )
    docs_test = _compute_doc_embeddings(
        test_texts,
        use_openrouter=use_openrouter,
        openrouter_model_name=openrouter_model_name,
    )
    if docs_val is None or docs_test is None:
        return None

    sim_val = docs_val @ centroids.T
    sim_test = docs_test @ centroids.T

    top1_idx_val, top2_idx_val, top1_scores_val, top2_scores_val = _top2_from_similarity(sim_val)
    top1_idx_test, top2_idx_test, top1_scores_test, top2_scores_test = _top2_from_similarity(sim_test)

    def rerank(i: int, p1: int, p2: int) -> int:
        return _keyword_rerank_index(val_texts[i], p1, p2, labels)

    baseline_val_preds = [labels[int(i)] for i in top1_idx_val]
    baseline_test_preds = [labels[int(i)] for i in top1_idx_test]
    baseline_val_f1 = _macro_f1(y_val, baseline_val_preds, labels)
    baseline_test_f1 = _macro_f1(y_test, baseline_test_preds, labels)

    thresholds = np.arange(0.0, 0.1001, 0.001)
    best_threshold = 0.0
    best_score = -1.0

    for thresh in thresholds:
        preds: list[str] = []
        for i in range(len(val_texts)):
            p1 = int(top1_idx_val[i])
            p2 = int(top2_idx_val[i])
            gap = float(top1_scores_val[i] - top2_scores_val[i])
            final_idx = rerank(i, p1, p2) if gap <= float(thresh) else p1
            preds.append(labels[final_idx])

        score = _macro_f1(y_val, preds, labels)
        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    pair_impact: dict[tuple[str, str], int] = {}
    for i in range(len(val_texts)):
        p1 = int(top1_idx_val[i])
        p2 = int(top2_idx_val[i])
        gap = float(top1_scores_val[i] - top2_scores_val[i])
        if gap <= best_threshold:
            continue

        gt = y_val[i]
        current = labels[p1]
        reranked = labels[rerank(i, p1, p2)]
        pair = (labels[p1], labels[p2])

        impact = 0
        if current != gt and reranked == gt:
            impact = 1
        elif current == gt and reranked != gt:
            impact = -1

        if impact != 0:
            pair_impact[pair] = pair_impact.get(pair, 0) + impact

    beneficial_pairs = {pair for pair, impact in pair_impact.items() if impact > 0}

    tuned_val_preds = _predict_with_rules(
        val_texts,
        labels,
        top1_idx_val,
        top2_idx_val,
        top1_scores_val,
        top2_scores_val,
        threshold=best_threshold,
        beneficial_pairs=beneficial_pairs,
    )
    tuned_test_preds = _predict_with_rules(
        test_texts,
        labels,
        top1_idx_test,
        top2_idx_test,
        top1_scores_test,
        top2_scores_test,
        threshold=best_threshold,
        beneficial_pairs=beneficial_pairs,
    )

    status: dict[str, Any] = {
        "tuned": True,
        "method": "validation_test_split",
        "split": {
            "validation_size": len(val_texts),
            "testing_size": len(test_texts),
            "test_size": 0.6,
            "random_state": 42,
        },
        "sources": bundle.get("sources", []),
        "label_column": bundle.get("label_column", "gt"),
        "threshold": best_threshold,
        "beneficial_pairs": [list(p) for p in sorted(beneficial_pairs)],
        "baseline": {
            "validation_macro_f1": baseline_val_f1,
            "testing_macro_f1": baseline_test_f1,
        },
        "tuned_scores": {
            "validation_macro_f1": _macro_f1(y_val, tuned_val_preds, labels),
            "testing_macro_f1": _macro_f1(y_test, tuned_test_preds, labels),
        },
    }
    return (best_threshold, beneficial_pairs, status)


def _resolve_rules(
    labels: list[str],
    *,
    use_openrouter: bool,
    openrouter_model_name: str | None,
) -> tuple[float, set[tuple[str, str]], dict[str, Any]]:
    key = _rules_cache_key(
        use_openrouter=use_openrouter,
        openrouter_model_name=openrouter_model_name,
    )
    if key in _RULES_CACHE:
        return _RULES_CACHE[key]

    default_threshold, default_pairs = _default_rules()
    model_name = _resolve_openrouter_model_name(openrouter_model_name) if use_openrouter else _load_model_name()
    status: dict[str, Any] = {
        "tuned": False,
        "method": "default_rules",
        "threshold": default_threshold,
        "beneficial_pairs": [list(p) for p in sorted(default_pairs)],
        "model": model_name,
    }

    # If model differs from default, try model-specific calibration.
    should_tune = False
    if use_openrouter:
        should_tune = _resolve_openrouter_model_name(openrouter_model_name) != _load_openrouter_model_name()

    if should_tune:
        try:
            tuned = _tune_rules_for_model(
                labels,
                use_openrouter=use_openrouter,
                openrouter_model_name=openrouter_model_name,
            )
            if tuned is not None:
                _RULES_CACHE[key] = tuned
                return tuned
            status["tuning_skipped"] = "insufficient_or_missing_calibration_data"
        except Exception as exc:  # noqa: BLE001
            status["tuning_error"] = str(exc)
            status["tuning_fallback"] = "default_rules"

    # Do not cache fallback results produced by transient tuning errors;
    # allow future requests to retry calibration.
    if "tuning_error" in status:
        return (default_threshold, set(default_pairs), status)

    _RULES_CACHE[key] = (default_threshold, set(default_pairs), status)
    return _RULES_CACHE[key]


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return set(tokens)


def _keyword_score(paragraph: str, label: str) -> float:
    lowered = paragraph.lower()
    score = 0.0
    for kw in _LABEL_KEYWORDS.get(label, []):
        if " " in kw:
            if kw in lowered:
                score += 1.0
        else:
            if kw in _tokenize(paragraph):
                score += 1.0
    return score


def _should_use_openrouter_embeddings(use_openrouter: bool | None) -> bool:
    if use_openrouter is not None:
        return use_openrouter

    backend = os.getenv("CLASSIFIER_EMBEDDINGS_BACKEND", "").strip().lower()
    return backend == "openrouter"


def _openrouter_embed_texts(texts: list[str], *, openrouter_model_name: str | None = None) -> np.ndarray:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter embeddings")

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model_name = _resolve_openrouter_model_name(openrouter_model_name)
    batch_size_raw = os.getenv("CLASSIFIER_OPENROUTER_BATCH_SIZE", "64").strip()
    try:
        batch_size = max(1, int(batch_size_raw))
    except ValueError:
        batch_size = 64

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = json.dumps(
            {
                "model": model_name,
                "input": batch,
                "encoding_format": "float",
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode("utf-8")
            raise RuntimeError(f"OpenRouter embeddings HTTP {exc.code}: {msg}") from exc

        vectors = [item.get("embedding", []) for item in body.get("data", [])]
        if not vectors:
            body_preview = str(body)[:300]
            raise RuntimeError(
                "OpenRouter embeddings returned empty data "
                f"for model '{model_name}' (batch start={start}, size={len(batch)}). "
                f"Response preview: {body_preview}"
            )

        all_vectors.extend(vectors)

    arr = np.array(all_vectors, dtype=np.float32)
    return arr


def _maybe_get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(sentence_transformers, "SentenceTransformer")
        _EMBED_MODEL = SentenceTransformer(_load_model_name())
        return _EMBED_MODEL
    except Exception:  # noqa: BLE001
        return None


def _compute_label_centroids(
    labels: list[str],
    *,
    use_openrouter: bool,
    openrouter_model_name: str | None = None,
) -> np.ndarray | None:
    global _CENTROIDS_CACHE, _OPENROUTER_CENTROIDS_CACHE

    model_key = _resolve_openrouter_model_name(openrouter_model_name) if use_openrouter else ""

    if use_openrouter:
        cached = _OPENROUTER_CENTROIDS_CACHE.get(model_key)
        if cached is not None and cached[0] == labels:
            return cached[1]
    else:
        if _CENTROIDS_CACHE is not None and _CENTROIDS_CACHE[0] == labels:
            return _CENTROIDS_CACHE[1]

    label_vectors: list[np.ndarray] = []
    for label in labels:
        phrases = _LABEL_KEYWORDS.get(label, [])
        if not phrases:
            phrases = [label]

        if use_openrouter:
            emb = _openrouter_embed_texts(
                phrases,
                openrouter_model_name=openrouter_model_name,
            )
        else:
            model = _maybe_get_embed_model()
            if model is None:
                return None
            emb = model.encode(phrases, convert_to_numpy=True)

        centroid = emb.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        label_vectors.append(centroid)

    centroids = np.stack(label_vectors)
    if use_openrouter:
        _OPENROUTER_CENTROIDS_CACHE[model_key] = (labels, centroids)
    else:
        _CENTROIDS_CACHE = (labels, centroids)
    return centroids


def _keyword_rerank_index(paragraph: str, top1_idx: int, top2_idx: int, labels: list[str]) -> int:
    top1_score = _keyword_score(paragraph, labels[top1_idx])
    top2_score = _keyword_score(paragraph, labels[top2_idx])
    if top2_score > top1_score:
        return top2_idx
    return top1_idx


def _softmax_confidence(scores: np.ndarray, top_idx: int) -> float:
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    denom = float(np.sum(exps))
    if denom <= 0.0:
        return 0.0
    return float(exps[top_idx] / denom)


def _classify_with_embeddings(
    paragraph: str,
    labels: list[str],
    *,
    use_openrouter: bool,
    openrouter_model_name: str | None,
    threshold: float,
    beneficial_pairs: set[tuple[str, str]],
) -> tuple[str, float, str | None, float | None, bool]:
    centroids = _compute_label_centroids(
        labels,
        use_openrouter=use_openrouter,
        openrouter_model_name=openrouter_model_name,
    )
    if centroids is None:
        raise RuntimeError("Embedding model unavailable")

    if use_openrouter:
        paragraph_vec = _openrouter_embed_texts(
            [paragraph],
            openrouter_model_name=openrouter_model_name,
        )[0]
    else:
        model = _maybe_get_embed_model()
        if model is None:
            raise RuntimeError("Embedding model unavailable")
        paragraph_vec = model.encode([paragraph], convert_to_numpy=True)[0]

    vec_norm = np.linalg.norm(paragraph_vec)
    if vec_norm > 0:
        paragraph_vec = paragraph_vec / vec_norm

    sim = centroids @ paragraph_vec
    order = np.argsort(sim)[::-1]
    top1_idx = int(order[0])
    top2_idx = int(order[1]) if len(order) > 1 else int(order[0])

    top1_score = float(sim[top1_idx])
    top2_score = float(sim[top2_idx])
    gap = top1_score - top2_score

    top_pair = (labels[top1_idx], labels[top2_idx])

    apply_rule = (gap <= threshold) or (top_pair in beneficial_pairs)
    final_idx = top1_idx
    if apply_rule:
        final_idx = _keyword_rerank_index(paragraph, top1_idx, top2_idx, labels)

    confidence = _softmax_confidence(sim, final_idx)

    return (
        labels[final_idx],
        confidence,
        labels[top2_idx],
        top2_score,
        apply_rule and (final_idx != top1_idx),
    )


def _classify_with_keywords(paragraph: str, labels: list[str]) -> tuple[str, float, str | None, float | None, bool]:
    scored = [(label, _keyword_score(paragraph, label)) for label in labels]
    scored.sort(key=lambda x: x[1], reverse=True)

    top1_label, top1_score = scored[0]
    top2_label, top2_score = scored[1] if len(scored) > 1 else (None, None)

    total = sum(max(s, 0.0) for _, s in scored)
    confidence = (top1_score / total) if total > 0 else 0.0
    return (top1_label, confidence, top2_label, top2_score, False)


def classify_paragraphs(
    paragraphs: list[str],
    *,
    source: str,
    use_openrouter_embeddings: bool | None = None,
    openrouter_embedding_model: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ArtifactClassification:
    """Classify paragraphs into EMC section labels.

    Parameters
    ----------
    paragraphs : list[str]
        Paragraph-level inputs only.
    source : str
        Source name, e.g. "paper", "model_card", "github_readme".
    use_openrouter_embeddings : bool | None
        ``True`` to force OpenRouter embeddings, ``False`` to force local
        sentence-transformers embeddings, ``None`` to use backend from
        ``CLASSIFIER_EMBEDDINGS_BACKEND``.
    openrouter_embedding_model : str | None
        Optional OpenRouter embedding model ID override, e.g.
        ``"qwen/qwen3-embedding-4b"``.
    """
    labels = _load_labels()
    use_openrouter = _should_use_openrouter_embeddings(use_openrouter_embeddings)
    threshold, beneficial_pairs, _ = _resolve_rules(
        labels,
        use_openrouter=use_openrouter,
        openrouter_model_name=openrouter_embedding_model,
    )

    paragraph_items = [
        (idx, paragraph)
        for idx, paragraph in enumerate(paragraphs)
        if isinstance(paragraph, str) and paragraph.strip()
    ]
    total = len(paragraph_items)
    source_label = source.replace("_", " ")

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "classification",
                "source": source,
                "completed": 0,
                "total": total,
                "remaining": total,
                "message": (
                    f"Embedding {source_label} paragraphs (0/{total} complete, {total} remaining)."
                    if total
                    else f"No paragraphs available for {source_label} classification."
                ),
            }
        )

    predictions: list[ParagraphPrediction] = []
    for position, (idx, paragraph) in enumerate(paragraph_items, start=1):

        try:
            top1_label, confidence, top2_label, top2_score, reranked = _classify_with_embeddings(
                paragraph,
                labels,
                use_openrouter=use_openrouter,
                openrouter_model_name=openrouter_embedding_model,
                threshold=threshold,
                beneficial_pairs=beneficial_pairs,
            )
        except Exception:  # noqa: BLE001
            top1_label, confidence, top2_label, top2_score, reranked = _classify_with_keywords(
                paragraph,
                labels,
            )

        predictions.append(
            ParagraphPrediction(
                paragraph=paragraph,
                predicted_label=top1_label,
                confidence=round(confidence, 4),
                top2_label=top2_label,
                top2_confidence=round(float(top2_score), 4) if top2_score is not None else None,
                source=source,
                index=idx,
                applied_rerank_rule=reranked,
            )
        )

        if progress_callback is not None:
            remaining = max(total - position, 0)
            progress_callback(
                {
                    "phase": "classification",
                    "source": source,
                    "completed": position,
                    "total": total,
                    "remaining": remaining,
                    "message": (
                        f"Embedding {source_label} paragraphs ({position}/{total} complete, {remaining} remaining)."
                        if total
                        else f"No paragraphs available for {source_label} classification."
                    ),
                }
            )

    if progress_callback is not None and total == 0:
        progress_callback(
            {
                "phase": "classification",
                "source": source,
                "completed": 0,
                "total": 0,
                "remaining": 0,
                "message": f"No paragraphs available for {source_label} classification.",
            }
        )

    return ArtifactClassification(source=source, predictions=predictions)


def classify_preprocessed_paragraphs(
    model_id: str,
    *,
    paper_paragraphs: list[str] | None = None,
    model_card_paragraphs: list[str] | None = None,
    github_paragraphs: list[str] | None = None,
    use_openrouter_embeddings: bool | None = None,
    openrouter_embedding_model: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ClassificationResult:
    """Classify paragraph splits for all available artifacts.

    This function is intentionally paragraph-only for Phase 1 integration.
    """
    labels = _load_labels()
    use_openrouter = _should_use_openrouter_embeddings(use_openrouter_embeddings)
    _, _, tuning_status = _resolve_rules(
        labels,
        use_openrouter=use_openrouter,
        openrouter_model_name=openrouter_embedding_model,
    )

    artifacts: list[tuple[str, list[str]]] = [
        ("paper", paper_paragraphs or []),
        ("model_card", model_card_paragraphs or []),
        ("github_readme", github_paragraphs or []),
    ]
    artifact_totals = {
        source: sum(1 for paragraph in paragraphs if isinstance(paragraph, str) and paragraph.strip())
        for source, paragraphs in artifacts
    }
    overall_total = sum(artifact_totals.values())

    def emit(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    if progress_callback is not None:
        emit(
            {
                "phase": "classification",
                "stage": "start",
                "completed": 0,
                "total": overall_total,
                "remaining": overall_total,
                "message": (
                    f"Embedding {overall_total} paragraphs for classification."
                    if overall_total
                    else "No paragraphs available for classification."
                ),
            }
        )

    completed_total = 0

    def wrap_progress(source: str, offset: int) -> Callable[[dict[str, Any]], None]:
        source_label = source.replace("_", " ")

        def _wrapped(event: dict[str, Any]) -> None:
            local_completed = int(event.get("completed", 0) or 0)
            overall_completed = min(overall_total, offset + local_completed)
            overall_remaining = max(overall_total - overall_completed, 0)
            emit(
                {
                    **event,
                    "phase": "classification",
                    "source": source,
                    "source_label": source_label,
                    "artifact_completed": local_completed,
                    "artifact_total": artifact_totals[source],
                    "artifact_remaining": max(artifact_totals[source] - local_completed, 0),
                    "completed": overall_completed,
                    "total": overall_total,
                    "remaining": overall_remaining,
                    "message": (
                        event.get("message")
                        or (
                            f"Embedding {source_label} paragraphs ({overall_completed}/{overall_total} complete, {overall_remaining} remaining)."
                            if overall_total
                            else f"No paragraphs available for {source_label} classification."
                        )
                    ),
                }
            )

        return _wrapped

    paper = classify_paragraphs(
        paper_paragraphs or [],
        source="paper",
        use_openrouter_embeddings=use_openrouter,
        openrouter_embedding_model=openrouter_embedding_model,
        progress_callback=wrap_progress("paper", completed_total),
    )
    completed_total += artifact_totals["paper"]

    model_card = classify_paragraphs(
        model_card_paragraphs or [],
        source="model_card",
        use_openrouter_embeddings=use_openrouter,
        openrouter_embedding_model=openrouter_embedding_model,
        progress_callback=wrap_progress("model_card", completed_total),
    )
    completed_total += artifact_totals["model_card"]

    github_readme = classify_paragraphs(
        github_paragraphs or [],
        source="github_readme",
        use_openrouter_embeddings=use_openrouter,
        openrouter_embedding_model=openrouter_embedding_model,
        progress_callback=wrap_progress("github_readme", completed_total),
    )

    if progress_callback is not None:
        emit(
            {
                "phase": "classification",
                "stage": "complete",
                "completed": overall_total,
                "total": overall_total,
                "remaining": 0,
                "message": "Classification complete.",
            }
        )

    success = bool(paper.predictions or model_card.predictions or github_readme.predictions)

    return ClassificationResult(
        model_id=model_id,
        labels=labels,
        paper=paper,
        model_card=model_card,
        github_readme=github_readme,
        tuning_status=tuning_status,
        success=success,
    )
