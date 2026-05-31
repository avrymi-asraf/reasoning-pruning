"""Run the automatic pruning-transition dataset creation loop.

This module is the core data-generation engine in the project architecture: for
each question it builds the context from previously accepted units, asks the
generator for a fresh reasoning continuation, splits that continuation into
units, asks the decision model for the first safe skip, builds a canonical PT
row, updates the pruned context with the useful prefix and target, and repeats.
It is designed for local uv smoke verification and for reuse in Hugging Face
Jobs that publish versioned datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from reasoning_pruning.data import build_pruning_transition_row
from reasoning_pruning.pruning_decision import PruningDecision
from reasoning_pruning.reasoning_units import split_reasoning_units
from reasoning_pruning.trace_generation import ReasoningGenerator


class PruningDecisionModel(Protocol):
    decision_model: str

    def find_first_removable_span(
        self, *, question: str, context: str, reasoning_units: list[str]
    ) -> PruningDecision:
        """Return the first conservative skip decision for generated units."""


@dataclass(frozen=True)
class DatasetBuildConfig:
    round_id: str
    max_pruning_depth: int = 1
    max_examples_per_question: int = 1
    unit_split_strategy: str = "numbered_or_lines"


def build_pt_dataset(
    *,
    questions: list[str],
    generator: ReasoningGenerator,
    decision_model: PruningDecisionModel,
    config: DatasetBuildConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        rows.extend(
            _build_rows_for_question(
                question=question,
                generator=generator,
                decision_model=decision_model,
                config=config,
            )
        )
    return rows


def _build_rows_for_question(
    *,
    question: str,
    generator: ReasoningGenerator,
    decision_model: PruningDecisionModel,
    config: DatasetBuildConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accepted_units: list[str] = []

    for depth in range(config.max_pruning_depth):
        if len(rows) >= config.max_examples_per_question:
            break

        context = _format_context(question, accepted_units)
        trace = generator.generate_reasoning(question=question, context=context)
        units = split_reasoning_units(trace.text, strategy=config.unit_split_strategy)
        if len(units) < 2:
            break

        decision = decision_model.find_first_removable_span(
            question=question,
            context=context,
            reasoning_units=units,
        )
        if not decision.valid_span_for(len(units)):
            break

        assert decision.removed_start_index is not None
        assert decision.removed_end_index is not None
        row = build_pruning_transition_row(
            question=question,
            context_before_generation=context,
            generated_trace=trace.text,
            generated_units=units,
            removed_start_index=decision.removed_start_index,
            removed_end_index=decision.removed_end_index,
            depth=depth,
            generator_model=generator.source_model,
            round_id=config.round_id,
            generator_model_revision=generator.source_model_revision,
            decision_model=decision_model.decision_model,
            decision_reason=decision.reason,
        )
        rows.append(row)

        accepted_units.extend(units[: decision.removed_start_index])
        accepted_units.append(units[decision.removed_end_index + 1])

    return rows


def _format_context(question: str, accepted_units: list[str]) -> str:
    if not accepted_units:
        return f"Question:\n{question}"
    return f"Question:\n{question}\n" + "\n".join(accepted_units)
