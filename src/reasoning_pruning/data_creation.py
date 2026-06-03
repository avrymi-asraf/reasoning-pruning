"""Create reasoning-pruning datasets from generated traces.

This module owns the full data-creation flow: load a YAML config, read source
questions, split generated reasoning into units, build `input_x -> target_y`
rows, and prepare or publish Hugging Face Dataset payloads. It is the small
center of the project that local CLI tools and HF Jobs call after constructing
model clients. It runs in uv-managed local checks and in remote dataset-creation
jobs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Protocol

import yaml


@dataclass(frozen=True)
class DataCreationConfig:
    round_id: str
    source_type: str
    source_dataset: str
    source_dataset_revision: str | None
    source_questions_path: str | None
    source_subset: str | None
    source_split: str
    source_question_field: str
    source_limit: int | None
    code_version: str | None
    hub_dataset_id: str
    private: bool
    generator: dict[str, Any]
    decision: dict[str, Any]
    generation: dict[str, Any]
    pruning: dict[str, Any]
    max_pruning_depth: int
    max_examples_per_question: int
    unit_split_strategy: str
    max_retries_per_depth: int
    max_units_per_batch: int


@dataclass(frozen=True)
class GeneratedTrace:
    text: str
    generation_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PruningDecision:
    has_removal: bool
    removed_start_index: int | None
    removed_end_index: int | None
    reason: str
    can_continue_after_skip: bool

    def valid_for(self, units: list[str]) -> bool:
        if not self.has_removal or not self.can_continue_after_skip:
            return False
        if self.removed_start_index is None or self.removed_end_index is None:
            return False
        if self.removed_start_index < 0 or self.removed_end_index < self.removed_start_index:
            return False
        return self.removed_end_index + 1 < len(units)


class ReasoningGenerator(Protocol):
    source_model: str
    source_model_revision: str | None

    def generate_reasoning(self, *, question: str, context: str) -> GeneratedTrace:
        ...


class PruningDecisionModel(Protocol):
    decision_model: str

    def find_first_removable_span(
        self, *, question: str, context: str, reasoning_units: list[str]
    ) -> PruningDecision:
        ...


HfLoader = Callable[..., Iterable[dict[str, Any]]]
_NUMBERED_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[\).\s-]+)")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]|[^.!?]+$")


def load_data_creation_config(path: Path) -> DataCreationConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    _require(raw, "round_id")
    _require(raw, "source_dataset")
    _require(raw, "hub_dataset_id")

    generator = _mapping(raw, "generator")
    decision = _mapping(raw, "decision")
    _require(generator, "model_id")
    _require(decision, "model_id")

    return DataCreationConfig(
        round_id=str(raw["round_id"]),
        source_type=str(raw.get("source_type", "local_file")),
        source_dataset=str(raw["source_dataset"]),
        source_dataset_revision=_optional_str(raw.get("source_dataset_revision")),
        source_questions_path=_optional_str(raw.get("source_questions_path")),
        source_subset=_optional_str(raw.get("source_subset")),
        source_split=str(raw.get("source_split", "train")),
        source_question_field=str(raw.get("source_question_field", "question")),
        source_limit=_optional_int(raw.get("source_limit")),
        code_version=_optional_str(raw.get("code_version")),
        hub_dataset_id=str(raw["hub_dataset_id"]),
        private=bool(raw.get("private", False)),
        generator=dict(generator),
        decision=dict(decision),
        generation=dict(_mapping(raw, "generation", default={})),
        pruning=dict(_mapping(raw, "pruning", default={})),
        max_pruning_depth=int(raw.get("max_pruning_depth", 1)),
        max_examples_per_question=int(raw.get("max_examples_per_question", 1)),
        unit_split_strategy=str(raw.get("unit_split_strategy", "numbered_or_lines")),
        max_retries_per_depth=_positive_int(raw.get("max_retries_per_depth", 3), "max_retries_per_depth"),
        max_units_per_batch=_minimum_int(raw.get("max_units_per_batch", 2), "max_units_per_batch", minimum=2),
    )


def load_questions(
    config: DataCreationConfig,
    *,
    hf_loader: HfLoader | None = None,
    hf_token: str | None = None,
) -> list[str]:
    if config.source_type == "local_file":
        if config.source_questions_path is None:
            return []
        path = Path(config.source_questions_path)
        questions = _load_local_questions(path, config.source_question_field)
        return _limited(questions, config.source_limit)

    if config.source_type == "hf_dataset":
        loader = hf_loader or _default_hf_loader()
        if config.source_subset:
            rows = loader(
                config.source_dataset,
                config.source_subset,
                split=config.source_split,
                revision=config.source_dataset_revision,
                token=hf_token,
            )
        else:
            rows = loader(
                config.source_dataset,
                split=config.source_split,
                revision=config.source_dataset_revision,
                token=hf_token,
            )
        questions = [_extract_question(row, config.source_question_field) for row in rows]
        return _limited([q for q in questions if q], config.source_limit)

    raise ValueError(f"unknown source_type: {config.source_type}")


def split_reasoning_units(text: str, *, strategy: str = "numbered_or_lines") -> list[str]:
    if strategy == "numbered_or_lines":
        units = [_clean_unit(line) for line in text.splitlines()]
        units = [unit for unit in units if unit]
        return units if len(units) > 1 else _split_sentences(text)
    if strategy == "lines":
        return [unit for unit in (_clean_unit(line) for line in text.splitlines()) if unit]
    if strategy == "sentences":
        return _split_sentences(text)
    raise ValueError(f"unknown reasoning unit split strategy: {strategy}")


def build_pt_dataset(
    *,
    questions: list[str],
    generator: ReasoningGenerator,
    decision_model: PruningDecisionModel,
    config: DataCreationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        rows.extend(build_rows_for_question(question=question, generator=generator, decision_model=decision_model, config=config))
    return rows


def format_context(question: str, accepted_units: list[str]) -> str:
    if not accepted_units:
        return f"Question:\n{question}"
    return f"Question:\n{question}\n" + "\n".join(accepted_units)


def build_pruning_transition_row(
    *,
    question: str,
    context_before_generation: str,
    generated_trace: str,
    generated_units: list[str],
    decision: PruningDecision,
    depth: int,
    generator_model: str,
    round_id: str,
    generator_model_revision: str | None = None,
    decision_model: str | None = None,
) -> dict[str, Any]:
    if not decision.valid_for(generated_units):
        raise ValueError("pruning decision does not point to a removable span with a following step")

    start = decision.removed_start_index
    end = decision.removed_end_index
    assert start is not None and end is not None

    useful_prefix = generated_units[:start]
    input_x = context_before_generation
    if useful_prefix:
        input_x += "\n" + "\n".join(useful_prefix)

    return {
        "id": _stable_row_id(round_id, question, depth, start, end),
        "question": question,
        "context_before_generation": context_before_generation,
        "generated_trace": generated_trace,
        "generated_units": generated_units,
        "input_x": input_x,
        "target_y": generated_units[end + 1],
        "pruning_depth": depth,
        "metadata": {
            "generator_model": generator_model,
            "generator_model_revision": generator_model_revision,
            "decision_model": decision_model,
            "removed_span": generated_units[start : end + 1],
            "removed_start_index": start,
            "removed_end_index": end,
            "decision_reason": decision.reason,
        },
    }


def advance_context_units(
    accepted_units: list[str],
    generated_units: list[str],
    decision: PruningDecision,
) -> list[str]:
    if not decision.valid_for(generated_units):
        raise ValueError("cannot advance context with an invalid pruning decision")
    start = decision.removed_start_index
    end = decision.removed_end_index
    assert start is not None and end is not None
    return [*accepted_units, *generated_units[:start], generated_units[end + 1]]


def transition_row_to_prompt_completion(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    converted["prompt"] = f"{row['input_x']}\n\nContinue with the next useful reasoning step:"
    converted["completion"] = row["target_y"]
    return converted


def prepare_hf_dataset_payload(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "canonical": rows,
        "training": [transition_row_to_prompt_completion(row) for row in rows],
    }


def push_pt_dataset_to_hub(
    *,
    rows: list[dict[str, Any]],
    hub_dataset_id: str,
    token: str | None,
    private: bool = False,
) -> str:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Publishing requires the optional 'datasets' package.") from exc

    payload = prepare_hf_dataset_payload(rows)
    Dataset.from_list(payload["canonical"]).push_to_hub(
        hub_dataset_id, config_name="canonical", split="train", token=token, private=private
    )
    Dataset.from_list(payload["training"]).push_to_hub(
        hub_dataset_id, config_name="training", split="train", token=token, private=private
    )
    return hub_dataset_id


def build_rows_for_question(
    *,
    question: str,
    generator: ReasoningGenerator,
    decision_model: PruningDecisionModel,
    config: DataCreationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accepted_units: list[str] = []

    for depth in range(config.max_pruning_depth):
        if len(rows) >= config.max_examples_per_question:
            break

        context = format_context(question, accepted_units)

        trace_out = None
        units_out = None
        decision_out = None
        attempts = 0

        for _ in range(config.max_retries_per_depth):
            attempts += 1
            trace = generator.generate_reasoning(question=question, context=context)
            units = split_reasoning_units(trace.text, strategy=config.unit_split_strategy)
            if not 2 <= len(units) <= config.max_units_per_batch:
                continue
            decision = decision_model.find_first_removable_span(
                question=question, context=context, reasoning_units=units
            )
            if decision.valid_for(units):
                trace_out, units_out, decision_out = trace, units, decision
                break

        if decision_out is None:
            break

        row = build_pruning_transition_row(
            question=question,
            context_before_generation=context,
            generated_trace=trace_out.text,
            generated_units=units_out,
            decision=decision_out,
            depth=depth,
            generator_model=generator.source_model,
            round_id=config.round_id,
            generator_model_revision=generator.source_model_revision,
            decision_model=decision_model.decision_model,
        )
        row["metadata"]["retry_attempts"] = attempts
        rows.append(row)

        accepted_units = advance_context_units(accepted_units, units_out, decision_out)
        assert format_context(question, accepted_units) == f"{row['input_x']}\n{row['target_y']}"

    return rows


def _load_local_questions(path: Path, question_field: str) -> list[str]:
    if path.suffix != ".jsonl":
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    questions: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if question_field not in row:
            raise ValueError(f"missing {question_field} field in {path}")
        question = str(row[question_field]).strip()
        if question:
            questions.append(question)
    return questions


def _extract_question(row: dict[str, Any], question_field: str) -> str:
    if question_field not in row:
        raise ValueError(f"missing {question_field} field in HF dataset row")
    return str(row[question_field]).strip()


def _limited(questions: list[str], limit: int | None) -> list[str]:
    return questions if limit is None else questions[:limit]


def _default_hf_loader() -> HfLoader:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install datasets to load source questions from Hugging Face") from exc
    return load_dataset


def _split_sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _clean_unit(text: str) -> str:
    return _NUMBERED_PREFIX_RE.sub("", text).strip()


def _stable_row_id(round_id: str, question: str, depth: int, start: int, end: int) -> str:
    payload = {
        "round_id": round_id,
        "question": question,
        "depth": depth,
        "removed_start_index": start,
        "removed_end_index": end,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"pt-{digest[:16]}"


def _require(raw: dict[str, Any], key: str) -> None:
    if not raw.get(key):
        raise ValueError(f"missing required data-creation config field: {key}")


def _mapping(raw: dict[str, Any], key: str, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    value = raw.get(key, default)
    if isinstance(value, dict):
        return value
    raise ValueError(f"{key} must be a mapping")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _positive_int(value: Any, field_name: str) -> int:
    return _minimum_int(value, field_name, minimum=1)


def _minimum_int(value: Any, field_name: str, *, minimum: int) -> int:
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return parsed


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
