"""Build canonical pruning-transition rows for the artifact flow.

This module is the row-builder layer in the project architecture: it turns a
question, the context passed to G, G's generated trace (with split units), and
the pruning decision into the canonical dataset contract. It is called by the
automatic PT dataset builder locally and in Hugging Face Jobs scripts before
rows are converted to TRL prompt/completion training examples.
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
    generator_model: str,
    round_id: str,
    context_before_generation: str | None = None,
    generator_model_revision: str | None = None,
    decision_model: str | None = None,
    decision_reason: str | None = None,
) -> dict[str, Any]:
    if removable_index < 0 or removable_index >= len(reasoning_steps):
        raise ValueError("removable_index is outside the reasoning path")
    if removable_index + 1 >= len(reasoning_steps):
        raise ValueError("removable step must have a following useful step")

    return build_pruning_transition_row(
        question=question,
        context_before_generation=context_before_generation or f"Question:\n{question}",
        generated_trace="\n".join(reasoning_steps),
        generated_units=reasoning_steps,
        removed_start_index=removable_index,
        removed_end_index=removable_index,
        depth=depth,
        generator_model=generator_model,
        round_id=round_id,
        generator_model_revision=generator_model_revision,
        decision_model=decision_model,
        decision_reason=decision_reason,
    )


def build_pruning_transition_row(
    *,
    question: str,
    context_before_generation: str,
    generated_trace: str,
    generated_units: list[str],
    removed_start_index: int,
    removed_end_index: int,
    depth: int,
    generator_model: str,
    round_id: str,
    generator_model_revision: str | None = None,
    decision_model: str | None = None,
    decision_reason: str | None = None,
) -> dict[str, Any]:
    if removed_start_index < 0 or removed_start_index >= len(generated_units):
        raise ValueError("removed_start_index is outside the reasoning path")
    if removed_end_index < removed_start_index or removed_end_index >= len(generated_units):
        raise ValueError("removed_end_index is outside the reasoning path")
    next_index = removed_end_index + 1
    if next_index >= len(generated_units):
        raise ValueError("removed span must have a following useful step")

    useful_prefix = generated_units[:removed_start_index]
    if useful_prefix:
        input_x = context_before_generation + "\n" + "\n".join(useful_prefix)
    else:
        input_x = context_before_generation

    return {
        "id": _stable_row_id(round_id, question, depth, removed_start_index, removed_end_index),
        "question": question,
        "context_before_generation": context_before_generation,
        "generated_trace": generated_trace,
        "generated_units": generated_units,
        "input_x": input_x,
        "target_y": generated_units[next_index],
        "pruning_depth": depth,
        "metadata": {
            "generator_model": generator_model,
            "generator_model_revision": generator_model_revision,
            "decision_model": decision_model,
            "removed_span": generated_units[removed_start_index : removed_end_index + 1],
            "removed_start_index": removed_start_index,
            "removed_end_index": removed_end_index,
            "decision_reason": decision_reason,
        },
    }


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
    generator_model: str = "google/gemma-4-E2B-it",
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
            generator_model=generator_model,
            round_id=round_id,
        )
        for index, item in enumerate(seeds)
    ]
