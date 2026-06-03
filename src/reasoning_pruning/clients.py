"""Model clients for reasoning-pruning data creation.

This module hides the live model details behind two small ideas: a generator G
continues reasoning from a context, and a decision model D chooses the first
safe removable span. The data-creation loop imports only these clients and never
knows tokenizer, REST, prompt, or JSON details. It runs in uv-managed local tools
and in Hugging Face Jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from reasoning_pruning.data_creation import GeneratedTrace, PruningDecision

GeminiTransport = Callable[[str, dict[str, Any]], dict[str, Any]]


class _NewlineStoppingCriteria:
    """Stops generation once max_units newline-terminated reasoning units are complete.

    Counts '\n' characters in the decoded generated text; each newline marks the end
    of one numbered reasoning step. Fires after the Nth newline so G never starts
    writing a (N+1)-th unit.
    """

    def __init__(self, tokenizer: Any, prompt_length: int, max_units: int) -> None:
        self._tokenizer = tokenizer
        self._prompt_length = prompt_length
        self._max_units = max_units

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        generated_ids = input_ids[0][self._prompt_length :]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.count("\n") >= self._max_units


@dataclass
class TransformersGenerator:
    source_model: str
    source_model_revision: str | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    max_units_per_batch: int = 2

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers generator requires the optional 'transformers' package.") from exc

        import torch

        token = os.environ.get("HF_TOKEN")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.source_model,
            revision=self.source_model_revision,
            token=token,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.source_model,
            revision=self.source_model_revision,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=token,
        )

    def generate_reasoning(self, *, question: str, context: str) -> GeneratedTrace:
        from transformers import StoppingCriteriaList

        prompt = _generator_prompt(self._tokenizer, context, self.max_units_per_batch)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_length = inputs["input_ids"].shape[-1]
        stopping_criteria = StoppingCriteriaList(
            [_NewlineStoppingCriteria(self._tokenizer, prompt_length, self.max_units_per_batch)]
        )
        output_ids = self._model.generate(**inputs, **self.generation_config, stopping_criteria=stopping_criteria)
        generated_ids = output_ids[0][prompt_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GeneratedTrace(text=text, generation_config=dict(self.generation_config))


@dataclass
class GeminiGenerator:
    source_model: str
    source_model_revision: str | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    max_units_per_batch: int = 2
    api_key_env: str = "GEMINI_API_KEY"
    transport: GeminiTransport | None = None

    def generate_reasoning(self, *, question: str, context: str) -> GeneratedTrace:
        prompt = _generator_instruction(context, self.max_units_per_batch)
        text = gemini_generate_text(
            model=self.source_model,
            prompt=prompt,
            generation_config=self.generation_config,
            api_key_env=self.api_key_env,
            transport=self.transport,
        )
        return GeneratedTrace(text=text.strip(), generation_config=dict(self.generation_config))


@dataclass
class TransformersDecisionModel:
    decision_model: str
    decision_config: dict[str, Any] = field(default_factory=dict)
    revision: str | None = None
    prompt_version: str = "incremental-skip-v2"
    prompts_dir: str = "prompts"

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers decision model requires the optional 'transformers' package.") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.decision_model, revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.decision_model,
            revision=self.revision,
            device_map="auto",
        )

    def find_first_removable_span(
        self, *, question: str, context: str, reasoning_units: list[str]
    ) -> PruningDecision:
        prompt = format_decision_prompt(
            question=question,
            context=context,
            reasoning_units=reasoning_units,
            prompt_version=self.prompt_version,
            prompts_dir=self.prompts_dir,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(**inputs, **self.decision_config.get("generation", {}))
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return parse_json_pruning_decision(self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip())


@dataclass
class GeminiDecisionModel:
    decision_model: str
    decision_config: dict[str, Any] = field(default_factory=dict)
    revision: str | None = None
    prompt_version: str = "incremental-skip-v2"
    prompts_dir: str = "prompts"
    api_key_env: str = "GEMINI_API_KEY"
    transport: GeminiTransport | None = None

    def find_first_removable_span(
        self, *, question: str, context: str, reasoning_units: list[str]
    ) -> PruningDecision:
        generation_config = dict(self.decision_config)
        generation_config["responseMimeType"] = "application/json"
        text = gemini_generate_text(
            model=self.decision_model,
            prompt=format_decision_prompt(
                question=question,
                context=context,
                reasoning_units=reasoning_units,
                prompt_version=self.prompt_version,
                prompts_dir=self.prompts_dir,
            ),
            generation_config=generation_config,
            api_key_env=self.api_key_env,
            transport=self.transport,
        )
        return parse_json_pruning_decision(text)


def create_generator_from_config(
    config: dict[str, Any], generation: dict[str, Any], max_units_per_batch: int = 2
):
    provider = config.get("provider", "transformers")
    if provider == "transformers":
        return TransformersGenerator(
            source_model=str(config["model_id"]),
            source_model_revision=config.get("revision"),
            generation_config=dict(generation),
            max_units_per_batch=max_units_per_batch,
        )
    if provider == "gemini":
        return GeminiGenerator(
            source_model=str(config["model_id"]),
            source_model_revision=config.get("revision"),
            generation_config=dict(generation),
            max_units_per_batch=max_units_per_batch,
            api_key_env=str(config.get("api_key_env", "GEMINI_API_KEY")),
        )
    raise ValueError(f"unsupported generator provider: {provider}")


def create_decision_model_from_config(
    config: dict[str, Any], pruning: dict[str, Any], prompts_dir: str = "prompts"
):
    provider = config.get("provider", "transformers-json")
    prompt_version = str(config.get("prompt_version", "incremental-skip-v2"))
    decision_config = dict(pruning)
    decision_config["prompt_version"] = prompt_version

    if provider == "transformers-json":
        return TransformersDecisionModel(
            decision_model=str(config["model_id"]),
            revision=config.get("revision"),
            prompt_version=prompt_version,
            prompts_dir=prompts_dir,
            decision_config=decision_config,
        )
    if provider == "gemini-json":
        return GeminiDecisionModel(
            decision_model=str(config["model_id"]),
            revision=config.get("revision"),
            prompt_version=prompt_version,
            prompts_dir=prompts_dir,
            decision_config=decision_config,
            api_key_env=str(config.get("api_key_env", "GEMINI_API_KEY")),
        )
    raise ValueError(f"unsupported decision provider: {provider}")


def gemini_generate_text(
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
    response = (transport or gemini_rest_transport)(url, body)
    return _extract_gemini_text(response)


def parse_json_pruning_decision(text: str) -> PruningDecision:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return PruningDecision(False, None, None, "Decision model did not return JSON.", False)
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return PruningDecision(False, None, None, "Decision model returned invalid JSON.", False)

    return PruningDecision(
        has_removal=bool(raw.get("has_removal", False)),
        removed_start_index=_optional_int(raw.get("removed_start_index")),
        removed_end_index=_optional_int(raw.get("removed_end_index")),
        reason=str(raw.get("reason", "")),
        can_continue_after_skip=bool(raw.get("can_continue_after_skip", False)),
    )


def format_decision_prompt(
    *,
    question: str,
    context: str,
    reasoning_units: list[str],
    prompt_version: str,
    prompts_dir: str = "prompts",
) -> str:
    template = load_prompt_template(prompt_version, prompts_dir)
    numbered_units = "\n".join(f"{index}: {unit}" for index, unit in enumerate(reasoning_units))
    return template.format(
        prompt_version=prompt_version,
        question=question,
        context=context,
        reasoning_units=numbered_units,
    )


def load_prompt_template(prompt_version: str, prompts_dir: str = "prompts") -> str:
    path = Path(prompts_dir) / f"{prompt_version}.txt"
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.read_text()


def gemini_rest_transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = _extract_error_message(detail) or detail[:500]
        raise RuntimeError(f"Gemini API error {exc.code}: {message}") from exc


def _generator_prompt(tokenizer: Any, context: str, max_units: int = 2) -> str:
    content = _generator_instruction(context, max_units)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def _generator_instruction(context: str, max_units: int) -> str:
    return (
        f"{context}\n\n"
        f"Continue the reasoning with exactly {max_units} numbered steps, one step per line. "
        "Do not write anything outside the numbered steps."
    )


def _extract_gemini_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    return "".join(str(part.get("text", "")) for part in content.get("parts") or [])


def _extract_error_message(detail: str) -> str:
    try:
        raw = json.loads(detail)
    except json.JSONDecodeError:
        return ""
    return str((raw.get("error") or {}).get("message", ""))


def _camelize_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "max_output_tokens": "maxOutputTokens",
        "top_p": "topP",
        "top_k": "topK",
        "response_mime_type": "responseMimeType",
    }
    ignored = {"prompt_version", "conservative", "require_following_step"}
    return {mapping.get(key, key): value for key, value in config.items() if key not in ignored and value is not None}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
