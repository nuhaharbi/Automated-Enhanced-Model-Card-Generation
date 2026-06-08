# Automated Enhanced Model Card Generation

Generate Enhanced Model Cards (EMCs) for Hugging Face models. The repository contains:

- A FastAPI web app for single-model generation.
- Shared pipeline components in `retriever/`, `preprocessor/`, `classifier/`, `evaluator/`, and `generator/`.
- The full dataset is maintained in a separate GitHub companion repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

If you only want the runtime dependencies, install the project itself or use the thin requirements wrapper:

```bash
python -m pip install -e .
# or
python -m pip install -r requirements.txt
```

## Web App Quick Start

### 1. Use the setup commands above

Follow the `Setup` section first if you have not already created the virtual environment and installed the dependencies.

### 2. Configure the environment variables

- `GITHUB_TOKEN` is required for GitHub API calls.
- `OPENROUTER_API_KEY` is required only if you want to use OpenRouter-backed embeddings or generator models. You can also enter the key directly in the web UI when you select the OpenRouter backend.
- `MODAL_EVALUATOR_URL` is required only if you want to run the reproducibility scorer without typing the endpoint into the web UI.

### 3. Provide an evaluator endpoint for reproducibility scoring

The web app does not ship with a private evaluator endpoint. Deploy your own evaluator and then paste its URL into the `Modal Evaluator Endpoint` field in the UI, or set `MODAL_EVALUATOR_URL`.

Use the included Modal app as a template:

```bash
modal deploy evaluator/modal_app.py
```

If your endpoint is protected, also set `MODAL_EVALUATOR_API_KEY`.

### 4. Run the web app

```bash
uvicorn webapp.app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

### 5. Browse the included sample cards

If you only want to view the samples already included in the app, start the web app and use the `Sample Selection` panel in the UI. That path lets you preview existing model cards without configuring the environment variables.

## Evaluator Endpoint

The scorer endpoint template lives in [evaluator/modal_app.py](evaluator/modal_app.py). Edit and redeploy that file if you want to change the model, the sampling settings, or the endpoint contract.

## Repository Layout

- `retriever/` fetches and normalizes raw artifacts.
- `preprocessor/` cleans model cards, papers, and GitHub READMEs. PDF parsing uses PyMuPDF by default so the pipeline stays CPU-friendly; Marker is not the default because it requires a GPU.
- `classifier/` assigns paragraph-level labels.
- `evaluator/` scores reproducibility.
- `generator/` generates the final Model Card.
- `webapp/` exposes the FastAPI single-model UI and endpoints.
- The companion dataset repository documents every dataset file, its schema, and how it is used.