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
        max_pruning_depth=2,
        max_examples_per_question=2,
        unit_split_strategy="sentences",
    )

    rows = build_pt_dataset(
        questions=["What is 2 + 3?"],
        generator=generator,
        decision_model=decision_model,
        config=config,
    )

    # context_before_generation for each depth
    assert rows[0]["context_before_generation"] == "Question:\nWhat is 2 + 3?"
    assert rows[1]["context_before_generation"] == "Question:\nWhat is 2 + 3?\nA.\nC."

    # invariant: input_x_d + "\n" + target_y_d == context_before_generation_{d+1}
    assert rows[0]["input_x"] + "\n" + rows[0]["target_y"] == rows[1]["context_before_generation"]

    # generated traces and units
    assert rows[0]["generated_trace"] == "A. B is unnecessary. C."
    assert rows[1]["generated_trace"] == "E. F is unnecessary. G."
    assert rows[0]["generated_units"] == ["A.", "B is unnecessary.", "C."]
    assert rows[1]["generated_units"] == ["E.", "F is unnecessary.", "G."]

    # input_x and target_y
    assert [row["input_x"] for row in rows] == [
        "Question:\nWhat is 2 + 3?\nA.",
        "Question:\nWhat is 2 + 3?\nA.\nC.\nE.",
    ]
    assert [row["target_y"] for row in rows] == ["C.", "G."]
    assert [row["pruning_depth"] for row in rows] == [0, 1]

    # contexts passed to the generator match the invariant
    assert generator.contexts == [
        "Question:\nWhat is 2 + 3?",
        "Question:\nWhat is 2 + 3?\nA.\nC.",
    ]

    # metadata exactly matches doc spec
    assert rows[0]["metadata"]["generator_model"] == "fake-generator"
    assert rows[0]["metadata"]["generator_model_revision"] == "rev-a"
    assert rows[0]["metadata"]["decision_model"] == "fake-decision"
    assert rows[0]["metadata"]["removed_span"] == ["B is unnecessary."]
    assert rows[0]["metadata"]["removed_start_index"] == 1
    assert rows[0]["metadata"]["removed_end_index"] == 1
    assert rows[0]["metadata"]["decision_reason"] == "Middle unit is redundant."


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
