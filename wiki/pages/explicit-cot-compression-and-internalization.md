---
title: "Explicit CoT Compression and Internalization"
type: comparison
status: developing
created: 2026-06-03
updated: 2026-06-03
aliases: [reasoning compression, chain-of-thought compression, implicit CoT, concise reasoning]
tags: [chain-of-thought, reasoning-efficiency, implicit-reasoning, reasoning-pruning]
source_pages: [from-explicit-cot-to-implicit-cot, reasoning-pruning-codebase, training-large-language-models-to-reason-in-a-continuous-latent-space, graph-based-chain-of-thought-pruning, beyond-a-better-planning-with-transformers]
related_pages: [reasoning-pruning-project]
---

# Explicit CoT Compression and Internalization

## Core Model

Reasoning-efficiency methods differ in **what they remove**, **where the missing computation is expected to go**, and **what interface remains at inference time**. “Shorter reasoning” is therefore not one method. It ranges from deleting semantically empty text while keeping useful explicit steps, through learning shorter but still visible trajectories, to eliminating natural-language intermediate reasoning entirely or replacing it with latent computation.

The key design question is not simply how many tokens can be removed. It is: **which information must remain externally represented for correctness, controllability, and auditability, and which information can safely be omitted or internalized?**

## Why It Matters

This distinction prevents a useful paper on implicit CoT from being applied in a way that breaks the reasoning-pruning project's contract. It also exposes a broader research ladder: the project can first learn to skip filler, then measure whether that creates a stable foundation for more aggressive compression, without prematurely assuming that essential reasoning should disappear.

## Dimensions of Comparison

| Approach | Removed content | Remaining inference interface | Where computation goes | Main benefit | Main risk |
| --- | --- | --- | --- | --- | --- |
| Selective local filler pruning | Semantically judged filler only | Concise explicit reasoning steps | Still largely in visible text, with learned jumps over filler | Interpretability and low-risk token reduction | Judge errors; limited savings if traces contain little filler |
| Global structural trace pruning | Weakly contributing branches or redundant reflection | Shorter explicit trajectory | A more efficient visible path | Larger trace reductions informed by dependencies | Graph or dependency errors; more complex pipeline |
| Stepwise Internalization | Progressively more CoT tokens, including useful reasoning | Final answer or partially shortened CoT | Model hidden states | No-CoT-like latency with CoT-trained behavior | Accuracy loss, instability, and loss of visible reasoning |
| Continuous latent reasoning | Natural-language intermediate decoding | Latent thought states plus final answer | Recycled continuous hidden states | Potentially richer, less language-constrained planning | New architecture/training complexity and low auditability |

## Explanation

### 1. Process supervision can be a scaffold rather than the final interface

Several sources support a common pattern: first expose the model to a detailed process, then train it toward a shorter one. Stepwise Internalization begins from explicit CoT and gradually removes it. Searchformer first learns A* search dynamics and later bootstraps shorter search traces. The reasoning-pruning project starts from G's own explicit reasoning and extracts transitions that skip judged filler. These methods disagree about the desired endpoint, but they all reject the assumption that direct final-answer training is always the best way to obtain efficient reasoning. Evidence: [[from-explicit-cot-to-implicit-cot]], [[beyond-a-better-planning-with-transformers]], [[reasoning-pruning-codebase]].

### 2. Selectivity and internalization solve different problems

The reasoning-pruning project asks whether the model can stop verbalizing content that adds no computation, deduction, or new fact. Stepwise Internalization asks whether the model can stop verbalizing intermediate reasoning at all. The former is a **semantic compression** problem; the latter is a **representation transfer** problem. A model may succeed at filler pruning without learning implicit reasoning, and a model may internalize useful steps even when those steps were not redundant. Evidence: [[reasoning-pruning-codebase]], [[from-explicit-cot-to-implicit-cot]].

### 3. Removal order is both a semantic and optimization issue

Stepwise Internalization reports that removing from the beginning works better than removing from the end, and that aggressive schedule changes destabilize training. The reasoning-pruning project independently chooses the first removable span and trains the next useful transition. This is not the same algorithm, but the alignment suggests a testable hypothesis: early safe skips may be easier to learn because the remaining continuation provides more positions and more downstream supervision. This is an inference, not an established result for this project. Evidence: [[from-explicit-cot-to-implicit-cot]], [[reasoning-pruning-codebase]].

### 4. Redundancy can be local or structural

A local filler unit can be identified without modeling the entire trace. Redundant reflection, however, may look reasonable sentence by sentence while adding no new dependency or repeatedly checking an established conclusion. Graph-based pruning treats this as a global structure problem. The methods are complementary: a conservative local judge is simpler and safer, while dependency-aware analysis may discover higher-value removals that local rules miss. Evidence: [[graph-based-chain-of-thought-pruning]], [[reasoning-pruning-codebase]].

### 5. Efficiency is a frontier, not a single target length

Stepwise Internalization explicitly reports intermediate accuracy–speed trade-offs, and Searchformer measures shorter search dynamics while preserving optimal solutions. The reasoning-pruning project should similarly evaluate a frontier across removal depth or learned checkpoints rather than only ask whether a dataset can be built. Token count, latency, answer correctness, and visible-step quality should be reported together. Evidence: [[from-explicit-cot-to-implicit-cot]], [[beyond-a-better-planning-with-transformers]].

### 6. Latent reasoning changes the meaning of a “step”

Coconut argues that a decoded word sequence may force premature commitment and that a continuous thought can represent multiple possible next steps. The reasoning-pruning project instead supervises one exact next textual unit from G. This contrast matters especially for planning and backtracking tasks: a concise textual path may still be a poor interface if the task benefits from maintaining multiple alternatives. Evidence: [[training-large-language-models-to-reason-in-a-continuous-latent-space]], [[reasoning-pruning-codebase]].

## Claims and Evidence

- **Claim:** Detailed process supervision followed by compression is a recurring strategy across natural-language reasoning and symbolic planning. Evidence: [[from-explicit-cot-to-implicit-cot]], [[beyond-a-better-planning-with-transformers]], [[reasoning-pruning-codebase]]. Status: supported.
- **Claim:** Selective filler pruning should not be described as implicit CoT, because useful reasoning remains explicitly generated. Evidence: [[reasoning-pruning-codebase]], [[from-explicit-cot-to-implicit-cot]]. Status: established.
- **Claim:** Gradual removal may be safer than abrupt objective changes when moving toward more compressed reasoning. Evidence: [[from-explicit-cot-to-implicit-cot]]. Status: supported within the paper's training setting.
- **Claim:** Dependency-aware pruning may complement local filler judgments by finding redundant reflection that is not obviously empty in isolation. Evidence: [[graph-based-chain-of-thought-pruning]]. Status: provisional for this project.
- **Claim:** Fully latent reasoning can have qualitatively different search behavior from a concise textual trace. Evidence: [[training-large-language-models-to-reason-in-a-continuous-latent-space]]. Status: supported within Coconut's studied tasks.

## Relationships

- **Generalizes →** [[reasoning-pruning-project]]: the project occupies the selective local filler-pruning region of the broader compression landscape. Evidence: [[reasoning-pruning-codebase]].
- **Contrasts with →** [[reasoning-pruning-project]]: full implicit and continuous latent reasoning remove the visible intermediate interface that the project currently preserves. Evidence: [[from-explicit-cot-to-implicit-cot]], [[training-large-language-models-to-reason-in-a-continuous-latent-space]], [[reasoning-pruning-codebase]].

## Boundaries and Failure Modes

- Fewer tokens do not by themselves prove better reasoning; shortening can delete necessary computation, reduce interpretability, or merely hide errors.
- A result on synthetic arithmetic, symbolic planning, or a specific logical benchmark should not be treated as evidence for broad generalization.
- The phrase “internalized reasoning” should be used cautiously when the evidence is behavioral rather than a direct account of hidden-state computation.
- The current project contract must not be weakened by allowing D to rewrite targets or by removing essential reasoning simply because other methods pursue more aggressive compression.

## Open Questions or Tensions

- How much of modern long-form reasoning is pure filler versus useful scratchpad computation versus redundant but semantically plausible reflection?
- Does learning to remove only filler create a stable curriculum toward safely skipping some derivable-but-useful steps, or are these separate capabilities?
- Are first-span removals easier for the current model to learn than equally safe later-span removals?
- On planning-heavy tasks, does concise explicit reasoning lose alternatives that a latent representation could preserve?
- What is the right joint metric for answer accuracy, emitted reasoning tokens, latency, and auditability?

## Sources

- [[from-explicit-cot-to-implicit-cot]] — establishes Stepwise Internalization, removal curricula, and the accuracy–speed frontier.
- [[reasoning-pruning-codebase]] — defines the project's selective local-transition contract.
- [[training-large-language-models-to-reason-in-a-continuous-latent-space]] — provides the continuous latent-reasoning endpoint and multi-path contrast.
- [[graph-based-chain-of-thought-pruning]] — adds global dependency-aware redundancy pruning.
- [[beyond-a-better-planning-with-transformers]] — supports process-first, compress-later reasoning in symbolic planning.
