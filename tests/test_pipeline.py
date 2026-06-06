"""End-to-end tests for the simple reasoning-pruning data creation flow.

These tests exercise the real PT loop with fake model clients so local checks
stay fast and deterministic. The optional live Gemini test is explicitly gated
by environment variables. The tests run under uv and validate the canonical
`input_x -> target_y` contract consumed by training.
"""

import json
import os
from pathlib import Path

import pytest

from reasoning_pruning.clients import (
    GeminiDecisionModel,
    create_decision_model_from_config,
    create_generator_from_config,
)
from reasoning_pruning.data_creation import (
    DataCreationConfig,
    GeneratedTrace,
    PruningDecision,
    build_pt_dataset,
    format_context,
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
        max_retries_per_depth=3,
        max_units_per_batch=2,
    )


class _FakeGenerator:
    source_model = "fake-g"
    source_model_revision: str | None = "v0"

    def generate_reasoning(self, *, question, context):
        return GeneratedTrace(
            text="1. We need to find the sum.\n2. 2 + 3 = 5.",
            generation_config={},
        )


class _FakeDecisionModel:
    decision_model = "fake-d"

    def find_first_removable_span(self, *, question, context, reasoning_units):
        if len(reasoning_units) < 2:
            return PruningDecision(False, None, None, "too few units", False)
        return PruningDecision(True, 0, 0, "filler", True)


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
    assert next_context == format_context(row["question"], [row["target_y"]])


def test_pipeline_discards_failed_attempts_and_retries_from_the_same_context():
    class _RetryingGenerator:
        source_model = "fake-g"
        source_model_revision = "v0"

        def __init__(self):
            self.contexts = []
            self.traces = iter(
                [
                    "1. Add the numbers.\n2. Calculate the result.",
                    "1. We need to find the sum.\n2. 2 + 3 = 5.",
                ]
            )

        def generate_reasoning(self, *, question, context):
            self.contexts.append(context)
            return GeneratedTrace(text=next(self.traces), generation_config={})

    class _SecondAttemptDecisionModel:
        decision_model = "fake-d"

        def find_first_removable_span(self, *, reasoning_units, **_):
            if reasoning_units[1] == "2 + 3 = 5.":
                return PruningDecision(True, 0, 0, "filler", True)
            return PruningDecision(False, None, None, "no useful target", False)

    generator = _RetryingGenerator()
    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=generator,
        decision_model=_SecondAttemptDecisionModel(),
        config=_config(),
    )

    assert len(rows) == 1
    assert generator.contexts == ["Question:\nWhat is 2 + 3?"] * 2
    assert rows[0]["generated_trace"] == "1. We need to find the sum.\n2. 2 + 3 = 5."
    assert rows[0]["generated_units"] == ["We need to find the sum.", "2 + 3 = 5."]
    assert rows[0]["metadata"]["retry_attempts"] == 2


def test_pipeline_rejects_batches_larger_than_the_configured_unit_limit():
    class _OversizedGenerator:
        source_model = "fake-g"
        source_model_revision = "v0"

        def __init__(self):
            self.calls = 0

        def generate_reasoning(self, **_):
            self.calls += 1
            return GeneratedTrace(text="1. Filler.\n2. 2 + 3 = 5.\n3. The answer is 5.", generation_config={})

    generator = _OversizedGenerator()
    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=generator,
        decision_model=_FakeDecisionModel(),
        config=_config(),
    )

    assert rows == []
    assert generator.calls == 3


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


def test_data_creation_config_loads_current_yaml():
    config = load_data_creation_config(Path("configs/data/dataset_builder_gsm8k_100_gemma4.yaml"))
    assert config.round_id == "gsm8k-gemma4-100-r2"
    assert config.generator["model_id"] == "avreymi/gemma-4-E2B-it-reasoning-pruning"


def test_gemini_decision_model_uses_structured_json_schema(monkeypatch):
    captured = {}

    def _transport(url, body):
        captured["body"] = body
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "has_removal": True,
                                        "removed_start_index": 0,
                                        "removed_end_index": 0,
                                        "reason": "The first unit is filler.",
                                        "can_continue_after_skip": True,
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    decision_model = GeminiDecisionModel(
        decision_model="gemini-test",
        decision_config={"temperature": 0.0, "max_output_tokens": 128},
        transport=_transport,
    )

    decision = decision_model.find_first_removable_span(
        question="What is 2 + 3?",
        context="Question:\nWhat is 2 + 3?",
        reasoning_units=["We need to solve it.", "2 + 3 = 5."],
    )

    generation_config = captured["body"]["generationConfig"]
    schema = generation_config["responseJsonSchema"]
    assert generation_config["responseMimeType"] == "application/json"
    assert schema["type"] == "object"
    assert set(schema["required"]) == {
        "has_removal",
        "removed_start_index",
        "removed_end_index",
        "reason",
        "can_continue_after_skip",
    }
    assert generation_config["temperature"] == 0.0
    assert generation_config["maxOutputTokens"] == 128
    assert decision.has_removal is True
    assert decision.removed_start_index == 0
    assert decision.removed_end_index == 0
    assert decision.can_continue_after_skip is True


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
