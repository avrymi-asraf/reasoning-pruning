"""End-to-end tests for the simple reasoning-pruning data creation flow.

These tests exercise the real PT loop with fake model clients so local checks
stay fast and deterministic. The optional live Gemini test is explicitly gated
by environment variables. The tests run under uv and validate the canonical
`input_x -> target_y` contract consumed by training.
"""

import os
from pathlib import Path

import pytest

from reasoning_pruning.clients import create_decision_model_from_config, create_generator_from_config
from reasoning_pruning.data_creation import (
    DataCreationConfig,
    GeneratedTrace,
    PruningDecision,
    build_pt_dataset,
    format_context,
    format_spectrum_question,
    load_data_creation_config,
    split_reasoning_units,
    transition_row_to_prompt_completion,
)


def _config(round_id: str = "test") -> DataCreationConfig:
    return DataCreationConfig(
        round_id=round_id,
        source_type="local_file",
        source_dataset="local",
        source_dataset_revision=None,
        source_questions_path=None,
        source_subset=None,
        source_split="train",
        source_question_field="question",
        source_limit=None,
        code_version=None,
        hub_dataset_id="local/test",
        private=False,
        generator={"provider": "fake", "model_id": "fake-g"},
        decision={"provider": "fake", "model_id": "fake-d"},
        generation={},
        pruning={},
        max_pruning_depth=1,
        max_examples_per_question=1,
        unit_split_strategy="numbered_or_lines",
    )


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
            return PruningDecision(False, None, None, "too few units", False)
        return PruningDecision(True, 1, 1, "filler", True)


def test_pipeline_produces_valid_training_rows():
    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=_FakeGenerator(),
        decision_model=_FakeDecisionModel(),
        config=_config(),
    )

    assert rows, "pipeline should produce at least one row"
    row = rows[0]
    assert row["input_x"] and row["target_y"]

    training = transition_row_to_prompt_completion(row)
    assert training["prompt"].endswith("Continue with the next useful reasoning step:")
    assert training["completion"] == row["target_y"]


def test_context_update_invariant_is_visible():
    row = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=_FakeGenerator(),
        decision_model=_FakeDecisionModel(),
        config=_config(),
    )[0]

    next_context = f"{row['input_x']}\n{row['target_y']}"
    assert next_context == format_context(row["question"], [row["generated_units"][0], row["target_y"]])


def test_pipeline_returns_empty_when_no_removal_found():
    class _NoRemoval:
        decision_model = "fake-d"

        def find_first_removable_span(self, **_):
            return PruningDecision(False, None, None, "", False)

    rows = build_pt_dataset(
        questions=["What is 2 + 3?", "What comes after 8?"],
        generator=_FakeGenerator(),
        decision_model=_NoRemoval(),
        config=_config(),
    )
    assert rows == []


def test_split_reasoning_units_numbered_and_sentence_fallback():
    assert split_reasoning_units("1. Add 2 + 3.\n2. The sum is 5.") == ["Add 2 + 3.", "The sum is 5."]
    assert split_reasoning_units("Add 2 + 3. The sum is 5.") == ["Add 2 + 3.", "The sum is 5."]


def test_spectrum_question_includes_context_and_choices_but_never_the_answer():
    row = {
        "input_mode": "question_with_context_and_choices",
        "question": "What regulates body processes?",
        "context": "The hypothalamus produces hormones that regulate body processes.",
        "choices": [{"label": "A", "text": "pancreas"}, {"label": "B", "text": "hypothalamus"}],
        "gold_answer": "hypothalamus",
        "gold_answer_label": "B",
        "reference_solution": "It is the hypothalamus because ...",
        "supporting_facts": "hypothalamus -> hormones",
    }
    text = format_spectrum_question(row)
    assert "The hypothalamus produces hormones" in text
    assert "(A) pancreas" in text and "(B) hypothalamus" in text
    assert "Options:" in text
    assert row["reference_solution"] not in text
    assert row["supporting_facts"] not in text


def test_spectrum_question_question_only_has_no_options_or_context():
    text = format_spectrum_question({"input_mode": "question_only", "question": "What is 2 + 2?"})
    assert text == "What is 2 + 2?"


def test_data_creation_config_loads_current_yaml():
    config = load_data_creation_config(Path("configs/data/dataset_builder_spectrum_gemma4.yaml"))
    assert config.round_id == "spectrum-gemma4-r2"
    assert config.source_dataset == "avreymi/reasoning-spectrum-qa"
    assert config.source_split == "data"
    assert config.generator["model_id"] == "avreymi/gemma-4-E2B-it-reasoning-pruning"


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run live LLM tests",
)
def test_live_gemini_pipeline_generates_reasoning_and_produces_rows():
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
        config=_config("live-test"),
    )
    assert isinstance(rows, list)
    for row in rows:
        assert row["input_x"] and row["target_y"]
        training = transition_row_to_prompt_completion(row)
        assert training["prompt"] and training["completion"]
