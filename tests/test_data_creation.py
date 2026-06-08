"""Behavior tests for the core pruning loop.

These check that the loop is wired and runs the canonical contract end to end:
it emits `input_x -> target_y` rows, holds the next-context invariant, gives up
cleanly when nothing is prunable, and discards bad attempts without polluting
the context. They intentionally do not assert internal strings, metadata field
shapes, or call counts — the real semantic check is the pipeline inspection
run in `src/reasoning_pruning/pipeline_inspection.py`.
"""

from reasoning_pruning.data_creation import (
    GeneratedTrace,
    build_pt_dataset,
    format_context,
    transition_row_to_prompt_completion,
)

from fakes import FakeDecisionModel, FakeGenerator, NoRemovalDecisionModel, make_config


def test_pipeline_produces_valid_rows_and_training_format():
    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=FakeGenerator(),
        decision_model=FakeDecisionModel(),
        config=make_config(),
    )

    assert rows, "pipeline should produce at least one row"
    row = rows[0]
    assert row["input_x"] and row["target_y"]

    training = transition_row_to_prompt_completion(row)
    assert training["completion"] == row["target_y"]


def test_next_context_invariant_holds():
    row = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=FakeGenerator(),
        decision_model=FakeDecisionModel(),
        config=make_config(),
    )[0]

    next_context = f"{row['input_x']}\n{row['target_y']}"
    assert next_context == format_context(row["question"], [row["target_y"]])


def test_pipeline_returns_empty_when_no_removal_found():
    rows = build_pt_dataset(
        questions=["What is 2 + 3?", "What comes after 8?"],
        generator=FakeGenerator(),
        decision_model=NoRemovalDecisionModel(),
        config=make_config(),
    )
    assert rows == []


def test_invalid_attempts_are_discarded_and_retried_from_the_same_context():
    class _RetryingGenerator:
        source_model = "fake-g"
        source_model_revision = "v0"

        def __init__(self):
            self.contexts = []
            self.traces = iter(
                [
                    "1. Too few units.\n2. Still too few.",  # rejected: < 3 units
                    "1. We need to find the sum.\n2. 2 + 3 = 5.\n3. The answer is 5.",
                ]
            )

        def generate_reasoning(self, *, context):
            self.contexts.append(context)
            return GeneratedTrace(text=next(self.traces))

    generator = _RetryingGenerator()
    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=generator,
        decision_model=FakeDecisionModel(),
        config=make_config(),
    )

    assert len(rows) == 1
    # The discarded attempt must not have polluted the context the retry sees.
    assert generator.contexts == ["Question:\nWhat is 2 + 3?"] * 2
