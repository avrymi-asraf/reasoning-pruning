---
title: "Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping"
type: source
source_kind: paper
author_or_origin: "Lucas Lehnert, Sainbayar Sukhbaatar, DiJia Su, Qinqing Zheng, Paul Mcvay, Michael Rabbat, and Yuandong Tian"
published: 2024-02-21
captured: 2026-06-03
url_or_location: "https://arxiv.org/abs/2402.14083"
reliability: primary
status: processed
tags: [planning, trace-compression, self-improvement, reasoning-efficiency]
---

# Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping

## Scope and Relevance

This paper studies tokenized search traces rather than natural-language CoT. It is relevant because it shows that learning from full process traces and then bootstrapping shorter successful traces can outperform direct plan prediction. It does not address semantic filler in language-model reasoning.

## Faithful Summary

Searchformer is first trained to predict A* search dynamics represented as token sequences. It is then fine-tuned to use fewer search steps while preserving optimal plans. The paper reports that Searchformer solves previously unseen Sokoban puzzles optimally 93.7% of the time while using up to 26.8% fewer search steps than the A* process used for initial training, and that it outperforms direct plan-prediction baselines with smaller models and less training data.

## Extracted Knowledge

- **Claim:** Full process supervision can be a useful scaffold even when the desired final behavior uses a shorter process.
  - Support: Searchformer is initialized from A* search dynamics before being trained toward shorter dynamics.
  - Scope/conditions: Demonstrated on symbolic planning tasks with explicit search traces.
  - Status: observed.
- **Claim:** Directly predicting the final plan can be less effective than first learning a detailed process and then shortening it.
  - Support: Reported comparisons against direct plan-prediction baselines.
  - Scope/conditions: The result should not be generalized beyond the paper's planning setup without evidence.
  - Status: observed.

## Limitations and Failure Modes

- Search traces have clearer state semantics and correctness checks than free-form natural-language reasoning.
- Shorter sampled search dynamics are not the same as removing only filler while copying the next exact G-written unit.

## Integration Candidates

- Support the process-first, compress-later pattern in [[explicit-cot-compression-and-internalization]].
- Inform evaluation ideas for [[reasoning-pruning-project]] around preserving correctness while reducing explicit steps.

## Tensions or Contradictions

- Searchformer can select globally shorter successful trajectories, while the current project intentionally makes conservative local edits.
