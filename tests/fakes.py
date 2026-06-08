"""Tiny fake G/D clients and a config builder shared across test files.

These satisfy the generator/decision-model protocols so the pipeline tests run
in milliseconds with no GPU or network. They are deliberately minimal: the real
"does the output make sense" check is the live pipeline inspection run, not
these fakes.
"""

from reasoning_pruning.data_creation import DataCreationConfig, GeneratedTrace, PruningDecision


def make_config(round_id: str = "test") -> DataCreationConfig:
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
        max_units_per_batch=4,
    )


class FakeGenerator:
    source_model = "fake-g"
    source_model_revision = "v0"

    def generate_reasoning(self, *, context):
        return GeneratedTrace(text="1. We need to find the sum.\n2. 2 + 3 = 5.\n3. The answer is 5.")


class FakeDecisionModel:
    """Removes the first unit (filler) and keeps the next as the target."""

    decision_model = "fake-d"

    def find_first_removable_span(self, *, question, context, reasoning_units):
        if len(reasoning_units) < 2:
            return PruningDecision(False, None, None, "too few units", False)
        return PruningDecision(True, 0, 0, "filler", True)


class NoRemovalDecisionModel:
    """Never finds anything prunable — drives the give-up / no-row path."""

    decision_model = "fake-d"

    def find_first_removable_span(self, **_):
        return PruningDecision(False, None, None, "nothing prunable", False)
