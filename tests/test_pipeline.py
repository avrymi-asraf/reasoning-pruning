"""End-to-end pipeline tests for reasoning-pruning data creation.

Three tests check what the project actually does: run the PT loop to produce
input_x/target_y training rows. Fake models are used for the fast path; the
live Gemini test requires explicit opt-in via env vars.
"""

import os

import pytest

from reasoning_pruning.data import transition_row_to_prompt_completion
from reasoning_pruning.pruning_decision import PruningDecision
from reasoning_pruning.pt_dataset_builder import DatasetBuildConfig, build_pt_dataset
from reasoning_pruning.trace_generation import GeneratedTrace


class _FakeGenerator:
    source_model = "fake-g"
    source_model_revision: str | None = "v0"

    def generate_reasoning(self, *, question, context):
        return GeneratedTrace(
            text="First, restate the problem. Now, this step is redundant filler. Finally, 2 + 3 = 5.",
            generation_config={},
        )


class _FakeDecisionModel:
    decision_model = "fake-d"

    def find_first_removable_span(self, *, question, context, reasoning_units):
        if len(reasoning_units) < 3:
            return PruningDecision(has_removal=False, removed_start_index=None, removed_end_index=None, reason="too few units", can_continue_after_skip=False)
        return PruningDecision(has_removal=True, removed_start_index=1, removed_end_index=1, reason="filler", can_continue_after_skip=True)


def test_pipeline_produces_valid_training_rows():
    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=_FakeGenerator(),
        decision_model=_FakeDecisionModel(),
        config=DatasetBuildConfig(round_id="test"),
    )

    assert rows, "pipeline should produce at least one row"
    row = rows[0]
    assert row["input_x"] and row["target_y"]

    training = transition_row_to_prompt_completion(row)
    assert training["prompt"].endswith("Continue with the next useful reasoning step:")
    assert training["completion"] == row["target_y"]


def test_pipeline_returns_empty_when_no_removal_found():
    class _NoRemoval:
        decision_model = "fake-d"

        def find_first_removable_span(self, **_):
            return PruningDecision(has_removal=False, removed_start_index=None, removed_end_index=None, reason="", can_continue_after_skip=False)

    rows = build_pt_dataset(
        questions=["What is 2 + 3?", "What comes after 8?"],
        generator=_FakeGenerator(),
        decision_model=_NoRemoval(),
        config=DatasetBuildConfig(round_id="test"),
    )
    assert rows == []


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run live LLM tests",
)
def test_live_gemini_pipeline_generates_reasoning_and_produces_rows():
    from reasoning_pruning.model_clients import create_decision_model_from_config, create_generator_from_config

    generator = create_generator_from_config(
        {"provider": "gemini", "model_id": "gemini-2.0-flash-lite"},
        {"maxOutputTokens": 512, "temperature": 0.7},
    )
    decision_model = create_decision_model_from_config(
        {"provider": "gemini-json", "model_id": "gemini-2.0-flash-lite", "prompt_version": "conservative-skip-v1"},
        {"temperature": 0.0},
    )

    trace = generator.generate_reasoning(
        question="If x + 2 = 5, what is x?",
        context="Question:\nIf x + 2 = 5, what is x?",
    )
    assert trace.text.strip(), "Gemini must return non-empty reasoning"

    rows = build_pt_dataset(
        questions=["If x + 2 = 5, what is x?"],
        generator=generator,
        decision_model=decision_model,
        config=DatasetBuildConfig(round_id="live-test"),
    )
    assert isinstance(rows, list)
    for row in rows:
        assert row["input_x"] and row["target_y"]
        training = transition_row_to_prompt_completion(row)
        assert training["prompt"] and training["completion"]
