"""Print qualitative traces for reasoning-pruning data creation.

This module mirrors the production pruning loop while exposing every local
transition for human inspection. It connects notebook experiments and script
runs to the same generator, decision model, and row-building functions used by
the CLI and HF Jobs paths. It is intended for cheap local/API-backed qualitative
checks before expensive dataset creation or training jobs.
"""

from __future__ import annotations

from typing import Any

from reasoning_pruning.data_creation import (
    DataCreationConfig,
    PruningDecision,
    PruningDecisionModel,
    ReasoningGenerator,
    advance_context_units,
    build_pruning_transition_row,
    format_context,
    split_reasoning_units,
)


def run_qualitative_pruning_inspection(
    *,
    question: str,
    generator: ReasoningGenerator,
    decision_model: PruningDecisionModel,
    config: DataCreationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accepted_units: list[str] = []

    print_section("Original question")
    print(question)

    for depth in range(config.max_pruning_depth):
        if len(rows) >= config.max_examples_per_question:
            break

        context = format_context(question, accepted_units)
        print_section(f"Depth {depth}")
        print_label("Context before generation", context)

        trace_out = None
        units_out = None
        decision_out = None
        attempts = 0

        for _ in range(config.max_retries_per_depth):
            attempts += 1
            trace = generator.generate_reasoning(context=context)
            units = split_reasoning_units(trace.text, strategy=config.unit_split_strategy)

            print_section(f"Depth {depth} / attempt {attempts}", marker="-")
            print_label("G generated reasoning trace", trace.text)
            print_units(units)

            if not 2 <= len(units) <= config.max_units_per_batch:
                print(
                    "Attempt rejected: "
                    f"expected 2..{config.max_units_per_batch} units, got {len(units)}."
                )
                continue

            decision = decision_model.find_first_removable_span(
                question=question,
                context=context,
                reasoning_units=units,
            )
            print_decision(decision, units)

            if decision.valid_for(units):
                trace_out, units_out, decision_out = trace, units, decision
                break

            print("Attempt rejected: D did not return a valid removable span with a following useful unit.")

        if decision_out is None:
            print_section(f"Stopping after depth {depth}", marker="-")
            print("No valid pruning transition was found for this depth.")
            break

        row = build_pruning_transition_row(
            question=question,
            context_before_generation=context,
            generated_trace=trace_out.text,
            generated_units=units_out,
            decision=decision_out,
            depth=depth,
            generator_model=generator.source_model,
            round_id=config.round_id,
            generator_model_revision=generator.source_model_revision,
            decision_model=decision_model.decision_model,
        )
        row["metadata"]["retry_attempts"] = attempts
        rows.append(row)

        print_transition_row(row)
        accepted_units = advance_context_units(accepted_units, units_out, decision_out)
        next_context = format_context(question, accepted_units)
        expected_context = f"{row['input_x']}\n{row['target_y']}"
        assert next_context == expected_context
        print_label("Next context used for following depth", next_context)

    print_section("Inspection summary")
    print(f"Created {len(rows)} qualitative training row(s).")
    return rows


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
