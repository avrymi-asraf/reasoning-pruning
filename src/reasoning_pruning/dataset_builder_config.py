"""Load dataset-builder workflow configuration for reasoning-pruning.

This module owns the config contract for the automatic PT dataset creation
side of the project architecture. It validates source question selection,
generator/decision model choices, pruning limits, and the output Hugging Face
Dataset repo before the CLI or an HF Jobs dataset-builder script starts the
model-generated pruning loop. It is intended for uv-managed local dry runs and
remote dataset creation jobs, separate from training configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetBuilderConfig:
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


def load_dataset_builder_config(path: Path) -> DatasetBuilderConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    _require(raw, "round_id")
    _require(raw, "source_dataset")
    _require(raw, "hub_dataset_id")

    generator = _require_mapping(raw, "generator")
    decision = _require_mapping(raw, "decision")
    generation = _as_mapping(raw.get("generation", {}), "generation")
    pruning = _as_mapping(raw.get("pruning", {}), "pruning")

    _require(generator, "model_id")
    _require(decision, "model_id")

    return DatasetBuilderConfig(
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
        generation=dict(generation),
        pruning=dict(pruning),
        max_pruning_depth=int(raw.get("max_pruning_depth", 1)),
        max_examples_per_question=int(raw.get("max_examples_per_question", 1)),
        unit_split_strategy=str(raw.get("unit_split_strategy", "numbered_or_lines")),
    )


def _require(raw: dict[str, Any], key: str) -> None:
    if not raw.get(key):
        raise ValueError(f"missing required dataset-builder config field: {key}")


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _as_mapping(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"{key} must be a mapping")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
