"""Build canonical pruning-transition rows for the artifact flow.

This module is the row-builder layer described in AGENTS.md: it turns a
question, generated reasoning units, and a pruning decision into the canonical
`input_x -> target_y` dataset contract. It is called by the automatic PT dataset
builder locally and can be reused in Hugging Face Jobs before rows are converted
to TRL prompt/completion examples.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def build_pruning_transition_example(
    *,
    question: str,
    reasoning_steps: list[str],
    removable_index: int,
    depth: int,
    source_model: str,
    round_id: str,
    original_trace: str | None = None,
    source_model_revision: str | None = None,
    source_dataset: str | None = None,
    source_dataset_revision: str | None = None,
    decision_model: str | None = None,
    decision_config: Any | None = None,
    generation_config: Any | None = None,
    pruning_config: Any | None = None,
    code_version: str | None = None,
) -> dict[str, Any]:
    if removable_index < 0 or removable_index >= len(reasoning_steps):
        raise ValueError("removable_index is outside the reasoning path")
    if removable_index + 1 >= len(reasoning_steps):
        raise ValueError("removable step must have a following useful step")

    return build_pruning_transition_row(
        question=question,
        reasoning_units=reasoning_steps,
        removed_start_index=removable_index,
        removed_end_index=removable_index,
        depth=depth,
        source_model=source_model,
        round_id=round_id,
        original_trace=original_trace,
        source_model_revision=source_model_revision,
        source_dataset=source_dataset,
        source_dataset_revision=source_dataset_revision,
        decision_model=decision_model,
        decision_config=decision_config,
        generation_config=generation_config,
        pruning_config=pruning_config,
        code_version=code_version,
    )


def build_pruning_transition_row(
    *,
    question: str,
    reasoning_units: list[str],
    removed_start_index: int,
    removed_end_index: int,
    depth: int,
    source_model: str,
    round_id: str,
    original_trace: str | None = None,
    source_model_revision: str | None = None,
    source_dataset: str | None = None,
    source_dataset_revision: str | None = None,
    decision_model: str | None = None,
    decision_config: Any | None = None,
    generation_config: Any | None = None,
    pruning_config: Any | None = None,
    code_version: str | None = None,
    decision_reason: str | None = None,
) -> dict[str, Any]:
    if removed_start_index < 0 or removed_start_index >= len(reasoning_units):
        raise ValueError("removed_start_index is outside the reasoning path")
    if removed_end_index < removed_start_index or removed_end_index >= len(reasoning_units):
        raise ValueError("removed_end_index is outside the reasoning path")
    next_index = removed_end_index + 1
    if next_index >= len(reasoning_units):
        raise ValueError("removed span must have a following useful step")

    prefix_steps = reasoning_units[:removed_start_index]
    removed_units = reasoning_units[removed_start_index : removed_end_index + 1]
    next_useful_step = reasoning_units[next_index]
    prefix = "\n".join(prefix_steps) if prefix_steps else "(empty)"
    input_x = f"Question:\n{question}\n\nUseful reasoning prefix:\n{prefix}"

    row = {
        "id": _stable_row_id(round_id, question, depth, removed_start_index, removed_end_index),
        "question": question,
        "original_trace": original_trace,
        "input_x": input_x,
        "target_y": next_useful_step,
        "pruning_depth": depth,
        "metadata": {
            "source_model": source_model,
            "source_model_revision": source_model_revision,
            "round_id": round_id,
            "source_dataset": source_dataset,
            "source_dataset_revision": source_dataset_revision,
            "decision_model": decision_model,
            "decision_config": decision_config,
            "removed_span": "\n".join(removed_units),
            "removed_start_index": removed_start_index,
            "removed_end_index": removed_end_index,
            "generation_config": generation_config,
            "pruning_config": pruning_config,
            "code_version": code_version,
        },
    }
    if decision_reason is not None:
        row["metadata"]["decision_reason"] = decision_reason
    return row


def transition_row_to_prompt_completion(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    converted["prompt"] = (
        f"{row['input_x']}\n\n"
        "Continue with the next useful reasoning step:"
    )
    converted["completion"] = row["target_y"]
    return converted


def _stable_row_id(
    round_id: str,
    question: str,
    depth: int,
    removed_start_index: int,
    removed_end_index: int,
) -> str:
    payload = {
        "round_id": round_id,
        "question": question,
        "depth": depth,
        "removed_start_index": removed_start_index,
        "removed_end_index": removed_end_index,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"pt-{digest[:16]}"


def smoke_transition_examples(
    *,
    source_model: str = "google/gemma-4-E2B-it",
    round_id: str = "smoke-r0",
) -> list[dict[str, Any]]:
    seeds = [
        {
            "question": "If x + 2 = 5, what is x?",
            "steps": [
                "Subtract 2 from both sides.",
                "This is a very common algebra pattern.",
                "The result is x = 3.",
            ],
        },
        {
            "question": "A box has 3 red balls and 2 blue balls. How many balls are there?",
            "steps": [
                "Add the counts of red and blue balls.",
                "The color names do not change the arithmetic.",
                "3 + 2 = 5, so there are 5 balls.",
            ],
        },
        {
            "question": "What is the next number after 8?",
            "steps": [
                "Count forward by one from 8.",
                "This follows the standard integer order.",
                "The next number is 9.",
            ],
        },
        {
            "question": "If all cats are mammals and Luna is a cat, what is Luna?",
            "steps": [
                "Use the rule that every cat is a mammal.",
                "The name Luna is only identifying the subject.",
                "Since Luna is a cat, Luna is a mammal.",
            ],
        },
    ]

    return [
        build_pruning_transition_example(
            question=item["question"],
            reasoning_steps=item["steps"],
            removable_index=1,
            depth=index,
            source_model=source_model,
            round_id=round_id,
            original_trace="\n".join(item["steps"]),
        )
        for index, item in enumerate(seeds)
    ]
