"""Tests for the automatic reasoning-pruning dataset loop.

These tests encode the core project architecture: a generator model produces
reasoning, a decision model marks the first safe skip, and the builder emits
canonical pruning-transition rows. They run locally with fakes under uv/pytest
so the research loop can be verified without launching Hugging Face Jobs.
"""

from reasoning_pruning.pruning_decision import PruningDecision
from reasoning_pruning.pt_dataset_builder import DatasetBuildConfig, build_pt_dataset
from reasoning_pruning.trace_generation import GeneratedTrace


class FakeGenerator:
    source_model = "fake-generator"
    source_model_revision = "rev-a"

    def __init__(self) -> None:
        self.contexts: list[str] = []
        self.traces = [
            GeneratedTrace(
                text="A. B is unnecessary. C.",
                generation_config={"max_new_tokens": 64},
            ),
            GeneratedTrace(
                text="E. F is unnecessary. G.",
                generation_config={"max_new_tokens": 64},
            ),
        ]

    def generate_reasoning(self, *, question: str, context: str) -> GeneratedTrace:
        self.contexts.append(context)
        return self.traces.pop(0)


class FakeDecisionModel:
    decision_model = "fake-decision"
    decision_config = "conservative-v1"

    def __init__(self) -> None:
        self.calls = 0

    def find_first_removable_span(
        self, *, question: str, context: str, reasoning_units: list[str]
    ) -> PruningDecision:
        self.calls += 1
        return PruningDecision(
            has_removal=True,
            removed_start_index=1,
            removed_end_index=1,
            reason="Middle unit is redundant.",
            can_continue_after_skip=True,
        )


def test_builds_iterative_pt_rows_from_generator_and_decision_model():
    generator = FakeGenerator()
    decision_model = FakeDecisionModel()
    config = DatasetBuildConfig(
        round_id="round-001",
        source_dataset="local/questions",
        source_dataset_revision="abc123",
        code_version="code-sha",
        max_pruning_depth=2,
        max_examples_per_question=2,
        unit_split_strategy="sentences",
        pruning_config={"conservative": True},
    )

    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=generator,
        decision_model=decision_model,
        config=config,
    )

    assert [row["input_x"] for row in rows] == [
        "Question:\nWhat is 2 + 3?\n\nUseful reasoning prefix:\nA.",
        "Question:\nWhat is 2 + 3?\n\nUseful reasoning prefix:\nA.\nC.\nE.",
    ]
    assert [row["target_y"] for row in rows] == ["C.", "G."]
    assert [row["pruning_depth"] for row in rows] == [0, 1]
    assert generator.contexts == [
        "Question:\nWhat is 2 + 3?",
        "Question:\nWhat is 2 + 3?\n\nReasoning so far:\nA.\nC.",
    ]
    assert rows[0]["metadata"]["source_model"] == "fake-generator"
    assert rows[0]["metadata"]["source_model_revision"] == "rev-a"
    assert rows[0]["metadata"]["source_dataset"] == "local/questions"
    assert rows[0]["metadata"]["source_dataset_revision"] == "abc123"
    assert rows[0]["metadata"]["code_version"] == "code-sha"
    assert rows[0]["metadata"]["decision_model"] == "fake-decision"
    assert rows[0]["metadata"]["decision_config"] == "conservative-v1"
    assert rows[0]["metadata"]["removed_span"] == "B is unnecessary."
    assert rows[0]["metadata"]["removed_start_index"] == 1
    assert rows[0]["metadata"]["removed_end_index"] == 1
    assert rows[0]["metadata"]["generation_config"] == {"max_new_tokens": 64}
    assert rows[0]["metadata"]["pruning_config"] == {"conservative": True}
    assert rows[0]["original_trace"] == "A. B is unnecessary. C."
    assert rows[1]["original_trace"] == "E. F is unnecessary. G."


def test_stops_when_decision_has_no_safe_removal():
    class NoRemovalDecisionModel(FakeDecisionModel):
        def find_first_removable_span(
            self, *, question: str, context: str, reasoning_units: list[str]
        ) -> PruningDecision:
            return PruningDecision(
                has_removal=False,
                removed_start_index=None,
                removed_end_index=None,
                reason="All units are needed.",
                can_continue_after_skip=False,
            )

    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=FakeGenerator(),
        decision_model=NoRemovalDecisionModel(),
        config=DatasetBuildConfig(round_id="round-001"),
    )

    assert rows == []
