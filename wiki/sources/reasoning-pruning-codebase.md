---
title: "Reasoning-Pruning Codebase and Data-Creation Contract"
type: source
source_kind: codebase
author_or_origin: "reasoning-pruning repository"
published: unknown
captured: 2026-06-03
url_or_location: "AGENTS.md; src/reasoning_pruning/data_creation.py; prompts/conservative-skip-v1.txt; configs/data/*.yaml; configs/train/*.yaml"
reliability: primary
status: processed
tags: [reasoning-pruning, self-distillation, chain-of-thought, dataset-creation]
---

# Reasoning-Pruning Codebase and Data-Creation Contract

## Scope and Relevance

This source records the repository's current operational contract so that literature connections do not accidentally redefine the project. It covers how pruning examples are built, what the decision model may do, how context advances, and the active configuration shape. It does not provide empirical evidence that the trained model is already faster or more accurate.

## Faithful Summary

The project creates supervised local transitions from traces produced by the current model family. For each question and pruning depth, generator **G** produces a reasoning trace from the current pruned context. The trace is split into units, and decision model **D** identifies the first removable filler span. D is a judge only: the training target is copied exactly from the first G-written unit after the removed span.

The canonical training relation is `input_x -> target_y`, where `input_x` is the old context plus useful units before the removed span and `target_y` is the next useful G-written reasoning unit. The next generation context is required to equal `input_x + "\n" + target_y`. The conservative prompt forbids removing computation, numeric values, deductions, or new facts, and requires the following target unit to contain actual reasoning rather than another goal or intent statement.

The code supports repeated depths, but the currently checked-in dataset-builder configurations set `max_pruning_depth: 1` and `max_examples_per_question: 1`. Active round-2 data configurations use the current fine-tuned Gemma-4 model as G across GSM8K, MATH-Hard, AQUA-RAT, and BBH logical deduction.

## Extracted Knowledge

- **Definition/Object:** Reasoning-pruning transition — a supervised example that teaches the model to jump from a context before filler directly to the next useful step it already generated.
- **Contract:** D may select indices but may not create, paraphrase, summarize, or repair `target_y`.
  - Support: Repository instructions, row-construction code, and the conservative decision prompt.
  - Status: established.
- **Contract:** A valid row requires a following useful unit, and the next context must exactly equal `input_x + "\n" + target_y`.
  - Support: `PruningDecision.valid_for`, `build_pruning_transition_row`, `advance_context_units`, and the assertion in `build_rows_for_question`.
  - Status: established.
- **Claim:** The project is selective explicit-CoT compression, not full implicit CoT.
  - Support: The target remains an explicit reasoning unit, while only filler is eligible for removal.
  - Status: established.
- **Claim:** The project is iterative self-distillation at the behavior level.
  - Support: G is required to be the current fine-tuned model and targets are copied from its own traces.
  - Status: established.
- **Current limitation:** The implementation supports multi-depth pruning, but active configs currently exercise only one depth and one example per question.
  - Support: `build_rows_for_question` loops over depth; checked-in dataset configs set both limits to one.
  - Status: established.

## Limitations and Failure Modes

- D's semantic judgment can be wrong; the code validates index shape and target existence but cannot itself prove that the removed span is harmless or the target is useful.
- The current training setup does not yet establish an accuracy–token or accuracy–latency frontier.
- The local-transition objective teaches skipping known filler, not internalizing essential computations or replacing explicit reasoning with latent computation.
- Single-depth active configs do not yet test whether repeated pruning remains coherent over longer trajectories.

## Integration Candidates

- Anchor [[reasoning-pruning-project]] so project-specific recommendations remain consistent with the data-creation contract.
- Contrast with full internalization and latent reasoning in [[explicit-cot-compression-and-internalization]].

## Tensions or Contradictions

- Literature that rewrites targets, removes essential reasoning, or optimizes only final-answer length may be related to the project goal but is not compatible with the current data-creation contract.
