"""Template Modal deployment entrypoint for reproducibility scoring with GPT-OSS-20B.

Use this file as a starting point for your own endpoint:
    modal deploy evaluator/modal_app.py

Expected POST JSON:
- model: str
- system_prompt: str
- prompts: list[str]
- response_schema: dict

Response:
- outputs: list[str]
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
from typing import Any

import modal

app = modal.App("emc-repro-evaluator")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("vllm>=0.10.0", "pydantic>=2.0")
)


with image.imports():
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except Exception:  # noqa: BLE001
        StructuredOutputsParams = None
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except Exception:  # noqa: BLE001
        GuidedDecodingParams = None


@app.cls(
    image=image,
    gpu="A10G",
    timeout=60 * 20,
    scaledown_window=60 * 10,
)
class ReproScorer:
    @modal.enter()
    def load(self) -> None:
        self._model_name = os.getenv("MODAL_EVALUATOR_MODEL", "openai/gpt-oss-20b")
        self._llm = LLM(
            model=self._model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
        )
        self._tokenizer = self._llm.get_tokenizer()

    @modal.method()
    def score(self, *, model: str, system_prompt: str, prompts: list[str], response_schema: dict[str, Any]) -> list[str]:
        # Keep a stable loaded model; ignore per-request model overrides for safety.
        _ = model

        chat_prompts: list[str] = []
        for prompt in prompts:
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            chat_prompts.append(
                self._tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        sampling_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": 64000,
        }

        # Preferred path: same style as experiment code
        # structured_outputs=StructuredOutputsParams(json=response_schema)
        if StructuredOutputsParams is not None:
            try:
                sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                    json=response_schema
                )
            except Exception:  # noqa: BLE001
                pass
        elif GuidedDecodingParams is not None:
            # Compatibility fallback for vLLM variants using guided decoding.
            try:
                sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                    json=response_schema
                )
            except Exception:  # noqa: BLE001
                pass

        sampling_params = SamplingParams(
            **sampling_kwargs,
        )

        outputs = self._llm.generate(chat_prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]


scorer = ReproScorer()


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=True)
def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model", "openai/gpt-oss-20b"))
    system_prompt = str(payload.get("system_prompt", ""))
    prompts = payload.get("prompts", [])
    response_schema = payload.get("response_schema", {})

    if not isinstance(prompts, list) or not prompts:
        return {"outputs": [], "error": "prompts must be a non-empty list"}

    if not isinstance(response_schema, dict) or not response_schema:
        return {"outputs": [], "error": "response_schema must be a non-empty object"}

    outputs = scorer.score.remote(
        model=model,
        system_prompt=system_prompt,
        prompts=[str(p) for p in prompts],
        response_schema=response_schema,
    )

    return {"outputs": outputs}
