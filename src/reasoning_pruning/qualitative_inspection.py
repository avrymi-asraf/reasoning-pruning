"""Print every stage of reasoning-pruning data creation for human inspection.

This module does NOT reimplement the pruning loop. It runs the production
`build_rows_for_question` loop with a printing `PruningObserver`, so what a
human reads here is exactly what the dataset-creation and HF Jobs paths run —
the two can never drift. It is the real post-change check: after editing the
pipeline, run it (or the matching notebook cell) and judge whether G's traces,
the unit split, and D's decisions actually make sense. Intended for cheap
local/API-backed inspection before expensive dataset creation or training.
"""

from __future__ import annotations

from typing import Any

from reasoning_pruning.data_creation import (
    DataCreationConfig,
    PruningDecision,
    PruningDecisionModel,
    PruningObserver,
    ReasoningGenerator,
    build_rows_for_question,
)


def run_qualitative_pruning_inspection(
    *,
    question: str,
    generator: ReasoningGenerator,
    decision_model: PruningDecisionModel,
    config: DataCreationConfig,
) -> list[dict[str, Any]]:
    return build_rows_for_question(
        question=question,
        generator=generator,
        decision_model=decision_model,
        config=config,
        observer=_PrintingObserver(),
    )


class _PrintingObserver(PruningObserver):
    def start(self, question: str) -> None:
        print_section("Original question")
        print(question)

    def depth(self, depth: int, context: str) -> None:
        print_section(f"Depth {depth}")
        print_label("Context before generation", context)

    def attempt(self, depth: int, attempt: int, trace: str, units: list[str]) -> None:
        print_section(f"Depth {depth} / attempt {attempt}", marker="-")
        print_label("G generated reasoning trace", trace)
        print_units(units)

    def rejected_unit_count(self, got: int, maximum: int) -> None:
        print(f"Attempt rejected: expected 3..{maximum} units, got {got}.")

    def rejected_no_span(self) -> None:
        print("Attempt rejected: D found no valid removable span with a following useful unit.")

    def decision(self, decision: PruningDecision, units: list[str]) -> None:
        print_decision(decision, units)

    def row(self, row: dict[str, Any]) -> None:
        print_transition_row(row)

    def advanced(self, next_context: str) -> None:
        print_label("Next context used for following depth", next_context)

    def stopped(self, depth: int) -> None:
        print_section(f"Stopping after depth {depth}", marker="-")
        print("No valid pruning transition was found for this depth.")

    def done(self, rows: list[dict[str, Any]]) -> None:
        print_section("Inspection summary")
        print(f"Created {len(rows)} qualitative training row(s).")


def print_section(title: str, *, marker: str = "=") -> None:
    line = marker * 70
    print(f"\n{line}\n{title}\n{line}")


def print_label(label: str, value: str) -> None:
    print(f"\n[{label}]\n{value}")


def print_units(units: list[str]) -> None:
    print("\n[Split reasoning units]")
    if not units:
        print("  <none>")
        return
    for index, unit in enumerate(units):
        print(f"  {index}: {unit}")


def print_decision(decision: PruningDecision, units: list[str]) -> None:
    print("\n[D pruning decision]")
    print(f"  has_removal: {decision.has_removal}")
    print(f"  removed_start_index: {decision.removed_start_index}")
    print(f"  removed_end_index: {decision.removed_end_index}")
    print(f"  can_continue_after_skip: {decision.can_continue_after_skip}")
    print(f"  valid_for_units: {decision.valid_for(units)}")
    print(f"  reason: {decision.reason}")

    if decision.valid_for(units):
        start = decision.removed_start_index
        end = decision.removed_end_index
        assert start is not None and end is not None
        print("\n[Removed sentence/span]")
        for unit in units[start : end + 1]:
            print(f"  {unit}")
        print("\n[Selected target sentence]")
        print(f"  {units[end + 1]}")


def print_transition_row(row: dict[str, Any]) -> None:
    print_section("Final input_x -> target_y training row", marker="-")
    print_label("input_x", row["input_x"])
    print_label("target_y", row["target_y"])
    print("\n[Row metadata]")
    print(f"  id: {row['id']}")
    print(f"  pruning_depth: {row['pruning_depth']}")
    print(f"  retry_attempts: {row['metadata'].get('retry_attempts')}")
    print(f"  generator_model: {row['metadata']['generator_model']}")
    print(f"  decision_model: {row['metadata']['decision_model']}")
    print(f"  removed_span: {row['metadata']['removed_span']}")
    print(f"  decision_reason: {row['metadata']['decision_reason']}")
