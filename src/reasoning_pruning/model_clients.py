"""Model clients for PT trace generation and pruning decisions.

This module is the live model boundary in the project architecture: it provides
config-created generator and decision-model clients that the automatic dataset
builder can call. Heavy Transformers dependencies and remote Gemini calls are
kept behind explicit providers so local uv tests can verify the control flow
without downloading models or calling external APIs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from reasoning_pruning.pruning_decision import PruningDecision
from reasoning_pruning.trace_generation import GeneratedTrace

GeminiTransport = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass
class TransformersReasoningGenerator:
    source_model: str
    source_model_revision: str | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers generator requires the optional 'transformers' package. "
                "Run through uv with model dependencies or submit to HF Jobs."
            ) from exc

        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.source_model,
            revision=self.source_model_revision,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.source_model,
            revision=self.source_model_revision,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def generate_reasoning(self, *, question: str, context: str) -> GeneratedTrace:
        user_content = (
            f"{context}\n\n"
            "Continue the reasoning. Write each step as a concrete computation, deduction, "
            "or fact. Do not write goal statements or intent — compute directly."
        )
        if getattr(self._tokenizer, "chat_template", None):
            prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = user_content
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(**inputs, **self.generation_config)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GeneratedTrace(text=text, generation_config=dict(self.generation_config))


@dataclass
class TransformersJsonDecisionModel:
    decision_model: str
    decision_config: dict[str, Any] = field(default_factory=dict)
    revision: str | None = None
    prompt_version: str = "conservative-skip-v1"
    prompts_dir: str = "prompts"

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers decision model requires the optional 'transformers' package. "
                "Run through uv with model dependencies or submit to HF Jobs."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.decision_model, revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.decision_model,
            revision=self.revision,
            device_map="auto",
        )

    def find_first_removable_span(
        self, *, question: str, context: str, reasoning_units: list[str]
    ) -> PruningDecision:
        prompt = _decision_prompt(
            question=question,
            context=context,
            reasoning_units=reasoning_units,
            prompt_version=self.prompt_version,
            prompts_dir=self.prompts_dir,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(**inputs, **self.decision_config.get("generation", {}))
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return parse_json_pruning_decision(text)


@dataclass
class GeminiReasoningGenerator:
    source_model: str
    source_model_revision: str | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    api_key_env: str = "GEMINI_API_KEY"
    transport: GeminiTransport | None = None

    def generate_reasoning(self, *, question: str, context: str) -> GeneratedTrace:
        prompt = (
            f"{context}\n\n"
            "Continue the reasoning with concise numbered steps. "
            "Do not summarize; write the next reasoning path only."
        )
        text = _gemini_generate_text(
            model=self.source_model,
            prompt=prompt,
            generation_config=self.generation_config,
            api_key_env=self.api_key_env,
            transport=self.transport,
        )
        return GeneratedTrace(text=text.strip(), generation_config=dict(self.generation_config))


@dataclass
class GeminiJsonDecisionModel:
    decision_model: str
    decision_config: dict[str, Any] = field(default_factory=dict)
    revision: str | None = None
    prompt_version: str = "conservative-skip-v1"
    prompts_dir: str = "prompts"
    api_key_env: str = "GEMINI_API_KEY"
    transport: GeminiTransport | None = None

    def find_first_removable_span(
        self, *, question: str, context: str, reasoning_units: list[str]
    ) -> PruningDecision:
        prompt = _decision_prompt(
            question=question,
            context=context,
            reasoning_units=reasoning_units,
            prompt_version=self.prompt_version,
            prompts_dir=self.prompts_dir,
        )
        generation_config = dict(self.decision_config)
        generation_config["responseMimeType"] = "application/json"
        text = _gemini_generate_text(
            model=self.decision_model,
            prompt=prompt,
            generation_config=generation_config,
            api_key_env=self.api_key_env,
            transport=self.transport,
        )
        return parse_json_pruning_decision(text)


def parse_json_pruning_decision(text: str) -> PruningDecision:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return PruningDecision(
            has_removal=False,
            removed_start_index=None,
            removed_end_index=None,
            reason="Decision model did not return JSON.",
            can_continue_after_skip=False,
        )
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return PruningDecision(
            has_removal=False,
            removed_start_index=None,
            removed_end_index=None,
            reason="Decision model returned invalid JSON.",
            can_continue_after_skip=False,
        )

    return PruningDecision(
        has_removal=bool(raw.get("has_removal", False)),
        removed_start_index=_optional_int(raw.get("removed_start_index")),
        removed_end_index=_optional_int(raw.get("removed_end_index")),
        reason=str(raw.get("reason", "")),
        can_continue_after_skip=bool(raw.get("can_continue_after_skip", False)),
    )


def create_generator_from_config(config: dict[str, Any], generation: dict[str, Any]):
    provider = config.get("provider", "transformers")
    if provider == "transformers":
        return TransformersReasoningGenerator(
            source_model=str(config["model_id"]),
            source_model_revision=config.get("revision"),
            generation_config=dict(generation),
        )
    if provider == "gemini":
        return GeminiReasoningGenerator(
            source_model=str(config["model_id"]),
            source_model_revision=config.get("revision"),
            generation_config=dict(generation),
            api_key_env=str(config.get("api_key_env", "GEMINI_API_KEY")),
        )
    raise ValueError(f"unsupported generator provider: {provider}")


def create_decision_model_from_config(
    config: dict[str, Any], pruning: dict[str, Any], prompts_dir: str = "prompts"
):
    provider = config.get("provider", "transformers-json")
    decision_config = dict(pruning)
    decision_config["prompt_version"] = config.get("prompt_version", "conservative-skip-v1")
    if provider == "transformers-json":
        return TransformersJsonDecisionModel(
            decision_model=str(config["model_id"]),
            revision=config.get("revision"),
            prompt_version=str(config.get("prompt_version", "conservative-skip-v1")),
            prompts_dir=prompts_dir,
            decision_config=decision_config,
        )
    if provider == "gemini-json":
        return GeminiJsonDecisionModel(
            decision_model=str(config["model_id"]),
            revision=config.get("revision"),
            prompt_version=str(config.get("prompt_version", "conservative-skip-v1")),
            prompts_dir=prompts_dir,
            decision_config=decision_config,
            api_key_env=str(config.get("api_key_env", "GEMINI_API_KEY")),
        )
    raise ValueError(f"unsupported decision provider: {provider}")


def _gemini_generate_text(
    *,
    model: str,
    prompt: str,
    generation_config: dict[str, Any],
    api_key_env: str,
    transport: GeminiTransport | None,
) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for Gemini provider")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": _camelize_generation_config(generation_config),
    }
    response = (transport or _gemini_rest_transport)(url, body)
    return _extract_gemini_text(response)


def _gemini_rest_transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = _extract_error_message(detail) or detail[:500]
        raise RuntimeError(f"Gemini API error {exc.code}: {message}") from exc


def _extract_error_message(detail: str) -> str:
    try:
        raw = json.loads(detail)
    except json.JSONDecodeError:
        return ""
    error_body = raw.get("error") or {}
    return str(error_body.get("message", ""))


def _extract_gemini_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    return "".join(str(part.get("text", "")) for part in parts)


def _camelize_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "max_output_tokens": "maxOutputTokens",
        "top_p": "topP",
        "top_k": "topK",
        "response_mime_type": "responseMimeType",
    }
    ignored = {"prompt_version", "conservative", "require_following_step"}
    return {
        mapping.get(key, key): value
        for key, value in config.items()
        if key not in ignored and value is not None
    }


def _load_prompt_template(prompt_version: str, prompts_dir: str = "prompts") -> str:
    path = Path(prompts_dir) / f"{prompt_version}.txt"
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.read_text()


def _decision_prompt(
    *,
    question: str,
    context: str,
    reasoning_units: list[str],
    prompt_version: str,
    prompts_dir: str = "prompts",
) -> str:
    template = _load_prompt_template(prompt_version, prompts_dir)
    numbered_units = "\n".join(f"{index}: {unit}" for index, unit in enumerate(reasoning_units))
    return template.format(
        prompt_version=prompt_version,
        question=question,
        context=context,
        reasoning_units=numbered_units,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
