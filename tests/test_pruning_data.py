"""Tests for the pruning-transition data contract.

These tests protect the core code flow from AGENTS.md by verifying that local
data helpers produce canonical input/target rows that can later be converted
for TRL training. They run locally under uv/pytest and provide fast feedback
before remote Hugging Face Jobs are launched.
"""

from reasoning_pruning.data import (
    build_pruning_transition_example,
    transition_row_to_prompt_completion,
)


def test_builds_transition_from_question_prefix_and_next_useful_step():
    steps = [
        "Subtract 2 from both sides.",
        "This is a very common algebra pattern.",
        "The result is x = 3.",
    ]
    example = build_pruning_transition_example(
        question="If x + 2 = 5, what is x?",
        reasoning_steps=steps,
        removable_index=1,
        depth=0,
        source_model="google/gemma-4-E2B-it",
        round_id="smoke-r0",
        original_trace="\n".join(steps),
    )

    assert example["input_x"] == (
        "Question:\nIf x + 2 = 5, what is x?\n\n"
        "Useful reasoning prefix:\nSubtract 2 from both sides."
    )
    assert example["target_y"] == "The result is x = 3."
    assert example["question"] == "If x + 2 = 5, what is x?"
    assert example["original_trace"] == "\n".join(steps)
    assert example["pruning_depth"] == 0
    assert example["metadata"]["source_model"] == "google/gemma-4-E2B-it"
    assert example["metadata"]["round_id"] == "smoke-r0"
    assert example["metadata"]["removed_span"] == "This is a very common algebra pattern."


def test_converts_canonical_transition_row_to_prompt_completion_for_training():
    steps = [
        "Subtract 2 from both sides.",
        "This is a very common algebra pattern.",
        "The result is x = 3.",
    ]
    example = build_pruning_transition_example(
        question="If x + 2 = 5, what is x?",
        reasoning_steps=steps,
        removable_index=1,
        depth=0,
        source_model="google/gemma-4-E2B-it",
        round_id="smoke-r0",
        original_trace="\n".join(steps),
    )

    converted = transition_row_to_prompt_completion(example)

    assert converted["prompt"] == (
        "Question:\nIf x + 2 = 5, what is x?\n\n"
        "Useful reasoning prefix:\nSubtract 2 from both sides.\n\n"
        "Continue with the next useful reasoning step:"
    )
    assert converted["completion"] == "The result is x = 3."
    assert converted["original_trace"] == "\n".join(steps)


def test_rejects_removal_without_following_useful_step():
    try:
        build_pruning_transition_example(
            question="Why?",
            reasoning_steps=["Only step."],
            removable_index=0,
            depth=0,
            source_model="google/gemma-4-E2B-it",
            round_id="smoke-r0",
            original_trace="Only step.",
        )
    except ValueError as exc:
        assert "following useful step" in str(exc)
    else:
        raise AssertionError("expected ValueError")
