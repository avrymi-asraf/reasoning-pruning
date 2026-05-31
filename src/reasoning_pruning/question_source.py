"""Load source questions for the automatic PT dataset loop.

This module is the source-data boundary in the project architecture: it reads
questions from simple local files or Hugging Face Datasets before the generator
model creates reasoning traces. It runs under uv locally for smoke/test formats
and can also run in Hugging Face Jobs with `datasets` available for source
datasets hosted on the Hub.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuestionSourceConfig:
    source_type: str
    source_dataset: str
    source_dataset_revision: str | None
    source_questions_path: str | None
    source_subset: str | None
    source_split: str
    source_question_field: str
    source_limit: int | None


HfLoader = Callable[..., Iterable[dict[str, Any]]]


def load_questions(path: Path, question_field: str = "question") -> list[str]:
    if path.suffix == ".jsonl":
        return _load_jsonl_questions(path, question_field=question_field)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_questions_from_source(
    config: QuestionSourceConfig,
    *,
    hf_loader: HfLoader | None = None,
    hf_token: str | None = None,
) -> list[str]:
    if config.source_type == "local_file":
        if config.source_questions_path is None:
            return []
        questions = load_questions(Path(config.source_questions_path), config.source_question_field)
        return _limit_questions(questions, config.source_limit)
    if config.source_type == "hf_dataset":
        questions = load_questions_from_hf_dataset(
            dataset_id=config.source_dataset,
            subset=config.source_subset,
            split=config.source_split,
            question_field=config.source_question_field,
            revision=config.source_dataset_revision,
            limit=config.source_limit,
            hf_loader=hf_loader,
            hf_token=hf_token,
        )
        return questions
    raise ValueError(f"unknown source_type: {config.source_type}")


def load_questions_from_hf_dataset(
    *,
    dataset_id: str,
    subset: str | None,
    split: str,
    question_field: str,
    revision: str | None,
    limit: int | None,
    hf_loader: HfLoader | None = None,
    hf_token: str | None = None,
) -> list[str]:
    loader = hf_loader or _default_hf_loader()
    if subset:
        rows = loader(dataset_id, subset, split=split, revision=revision, token=hf_token)
    else:
        rows = loader(dataset_id, split=split, revision=revision, token=hf_token)
    questions = [_extract_question(row, question_field) for row in rows]
    return _limit_questions([question for question in questions if question], limit)


def _load_jsonl_questions(path: Path, question_field: str) -> list[str]:
    questions: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if question_field not in row:
            raise ValueError(f"missing {question_field} field in {path}")
        questions.append(str(row[question_field]).strip())
    return [question for question in questions if question]


def _extract_question(row: dict[str, Any], question_field: str) -> str:
    if question_field not in row:
        raise ValueError(f"missing {question_field} field in HF dataset row")
    return str(row[question_field]).strip()


def _limit_questions(questions: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return questions
    return questions[:limit]


def _default_hf_loader() -> HfLoader:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install datasets to load source questions from Hugging Face") from exc
    return load_dataset
