---
title: "Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs"
type: source
source_kind: paper
author_or_origin: "Hongyuan Yuan, Xinran He, Run Shao, Bolei He, Xianwei Xue, Mengke Chen, Qiutong Pan, Haiwei Wang, and Haifeng Li"
published: 2026-04-07
captured: 2026-06-03
url_or_location: "https://arxiv.org/abs/2604.05643"
reliability: primary
status: processed
tags: [chain-of-thought, reasoning-pruning, redundancy, reasoning-efficiency]
---

# Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs

## Scope and Relevance

This paper targets redundant reflection in long reasoning-model traces. It is relevant because it treats redundancy as a structural dependency problem and distills pruned behavior back into a model. It differs from the reasoning-pruning project in its trace representation, optimization pipeline, and willingness to optimize whole concise trajectories rather than exact local next-step transitions.

## Faithful Summary

The authors describe two forms of inefficient reflection: broad low-impact checking and repeated re-verification of an already established conclusion. Their framework converts a linear CoT into a directed acyclic graph with explicit dependency edges, then applies branch-level and depth-level pruning. The pruned behavior is distilled with a three-stage SFT, DPO, and GRPO pipeline. The paper reports a 42% reduction in average reasoning tokens while maintaining or improving accuracy.

## Extracted Knowledge

- **Definition/Object:** Redundant reflection — reasoning content that checks broadly with low impact or re-verifies conclusions that are already established.
- **Method:** Dependency-aware CoT pruning — represent a trace as a DAG and remove weakly contributing branches or late-stage re-verification.
- **Claim:** Structural dependency information can support substantial reasoning-token reduction without sacrificing accuracy.
  - Support: Reported experiments show a 42% average token reduction with maintained or improved accuracy.
  - Scope/conditions: The result belongs to the paper's models, benchmarks, graph construction, and optimization pipeline.
  - Status: observed.

## Limitations and Failure Modes

- Constructing reliable dependency graphs is itself a difficult judgment problem.
- The multi-stage preference and reinforcement-learning pipeline is substantially more complex than supervised local-transition extraction.
- The source does not establish that every local filler judgment should be replaced by graph-level pruning.

## Integration Candidates

- Add dependency-aware judging as a possible future investigation in [[reasoning-pruning-project]].
- Use the paper to distinguish local semantic pruning from global structural pruning in [[explicit-cot-compression-and-internalization]].

## Tensions or Contradictions

- The reasoning-pruning project preserves targets exactly from G and uses D only as an index judge; graph-based pruning optimizes broader trajectories through multiple training objectives.
