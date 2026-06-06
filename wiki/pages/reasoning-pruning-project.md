---
title: "Reasoning-Pruning Project"
type: entity
status: developing
created: 2026-06-03
updated: 2026-06-03
aliases: [reasoning pruning, local-transition reasoning pruning]
tags: [reasoning-pruning, self-distillation, chain-of-thought, research-roadmap]
source_pages: [reasoning-pruning-codebase, from-explicit-cot-to-implicit-cot, graph-based-chain-of-thought-pruning, beyond-a-better-planning-with-transformers, training-large-language-models-to-reason-in-a-continuous-latent-space]
related_pages: [explicit-cot-compression-and-internalization]
---

# Reasoning-Pruning Project

## Core Model

The reasoning-pruning project trains a model to skip its own non-useful verbal transitions while preserving the next useful reasoning step exactly as the model originally wrote it. A canonical example is:

```text
question + accepted useful prefix -> next useful G-written reasoning unit
```

Generator G produces the trace, decision model D selects a removable filler span, and the target is copied from the unit immediately after that span. The project therefore compresses explicit CoT without asking D to rewrite reasoning and without assuming that useful reasoning should become latent.

## Why It Matters

This formulation isolates a conservative research question: can a model learn to stop emitting habits such as restating goals, narrating intent, or adding empty commentary, while retaining its actual computations and deductions? The narrowness is a strength because it supports traceability and reduces the risk that “efficiency” comes from replacing the model's reasoning with a teacher's answer or an opaque rewrite.

## Current Method

1. Build the current context from the question and previously accepted useful units.
2. Ask the current fine-tuned model G to continue reasoning from that context.
3. Split G's exact trace into units.
4. Ask D for the first span that is pure filler and whose following unit is actual reasoning.
5. Emit `input_x -> target_y`, where `target_y` is copied exactly from G.
6. Advance the next context to exactly `input_x + "\n" + target_y`.
7. Repeat only while valid safe skips remain and configured limits allow it.

Evidence: [[reasoning-pruning-codebase]].

## Connections to Prior Work

### Stepwise Internalization: a neighboring objective, not the same objective

Both the project and Stepwise Internalization train models on contexts from which some earlier CoT text has been removed. Both also favor removal from the beginning of a continuation rather than starting with late-stage deletion. The critical difference is what may be removed: Stepwise Internalization progressively removes useful CoT tokens to make reasoning implicit, while this project removes only filler and keeps the next useful step explicit. Evidence: [[from-explicit-cot-to-implicit-cot]], [[reasoning-pruning-codebase]].

**Project implication:** use Stepwise Internalization as evidence that removal order, curriculum speed, and intermediate efficiency checkpoints deserve measurement, but do not import its full-removal objective into the current dataset contract.

### Searchformer: learn the process before shortening it

Searchformer shows a process-first, compress-later pattern in symbolic planning: detailed A* dynamics provide a scaffold, then shorter successful dynamics are learned. The project follows a related intuition with on-policy natural-language traces, but makes conservative local edits rather than selecting globally shorter trajectories. Evidence: [[beyond-a-better-planning-with-transformers]], [[reasoning-pruning-codebase]].

**Project implication:** compare against direct concise-answer or direct concise-reasoning baselines. If local-transition training wins, it would support the hypothesis that detailed process traces are useful scaffolds even when they are not the desired final output.

### Graph-based CoT pruning: local judgments versus global dependencies

The project's D prompt can detect obvious local filler, but repeated verification may appear reasonable in isolation. Graph-based pruning suggests that some redundancy is only visible after modeling dependencies across the full trace. Evidence: [[graph-based-chain-of-thought-pruning]], [[reasoning-pruning-codebase]].

**Project implication:** a future judge-analysis experiment could label whether rejected or missed removals require dependency information. This should initially be an evaluation tool, not a reason to let D rewrite targets or weaken exact-copy guarantees.

### Coconut: a different long-term endpoint

Coconut replaces textual intermediate reasoning with continuous latent thoughts and reports planning behavior that can preserve multiple alternatives. The project instead learns a single exact textual next step. Evidence: [[training-large-language-models-to-reason-in-a-continuous-latent-space]], [[reasoning-pruning-codebase]].

**Project implication:** planning-heavy benchmarks such as BBH logical deduction are important because they may reveal the boundary of concise linear explicit reasoning. A failure there would not necessarily mean pruning is useless; it may show that some tasks need a richer reasoning interface.

## Research Priorities Suggested by the Connections

### 1. Measure an accuracy–efficiency frontier

Evaluate checkpoints or datasets with different pruning depth and removal frequency. Report answer accuracy, emitted reasoning units or tokens, latency, and the fraction of traces that still contain judged filler. Stepwise Internalization shows why intermediate points can be valuable even when the most aggressive compression fails. Evidence: [[from-explicit-cot-to-implicit-cot]].

### 2. Exercise multi-depth pruning cautiously

The library supports repeated depth, but active configs currently cap each question at one pruning example. A controlled multi-depth experiment is the most direct way to test the project's central invariant over longer trajectories. Increase depth gradually and monitor whether later contexts remain coherent rather than jumping immediately to aggressive compression. Evidence: [[reasoning-pruning-codebase]], [[from-explicit-cot-to-implicit-cot]].

### 3. Separate judge quality from learner quality

Build an audit set that independently evaluates: whether D selected true filler, whether the following target is actual reasoning, whether the pruned context remains coherent, and whether the trained model learns the jump. This distinguishes data-label errors from optimization failures. Evidence: [[reasoning-pruning-codebase]], [[graph-based-chain-of-thought-pruning]].

### 4. Compare local and structural redundancy

Classify filler, restatement, intent, repeated verification, broad low-impact checking, and genuinely necessary scratchpad steps. Local filler may be addressable by the current prompt, while redundant reflection may need dependency-aware analysis. Evidence: [[graph-based-chain-of-thought-pruning]], [[reasoning-pruning-codebase]].

### 5. Test process-first compression against direct concise training

Include baselines that train on final answers only and on manually or mechanically shortened traces. This tests whether G's full trace provides useful supervision before the model learns to skip parts of it. Evidence: [[beyond-a-better-planning-with-transformers]], [[from-explicit-cot-to-implicit-cot]].

### 6. Preserve the current contract while exploring a separate compression ladder

A future research branch could ask whether some useful-but-derivable steps can be safely skipped. That would be a new objective requiring new validity criteria, stronger evaluation, and likely a curriculum. It must not be silently mixed into the current filler-only dataset because doing so would make results hard to interpret. Evidence: [[reasoning-pruning-codebase]], [[from-explicit-cot-to-implicit-cot]].

## Claims and Evidence

- **Claim:** The project is behavior-level on-policy self-distillation for concise explicit reasoning. Evidence: [[reasoning-pruning-codebase]]. Status: established.
- **Claim:** The project is closer to selective CoT compression than to implicit CoT. Evidence: [[reasoning-pruning-codebase]], [[from-explicit-cot-to-implicit-cot]]. Status: established.
- **Claim:** Gradual increases in compression depth are a reasonable experimental strategy because removal curricula can destabilize training. Evidence: [[from-explicit-cot-to-implicit-cot]]. Status: provisional for this project.
- **Claim:** Global dependency analysis may identify redundancy that a local filler judge misses. Evidence: [[graph-based-chain-of-thought-pruning]]. Status: provisional for this project.
- **Claim:** Planning-heavy tasks may expose limits of a single concise textual path. Evidence: [[training-large-language-models-to-reason-in-a-continuous-latent-space]]. Status: provisional for this project.

## Relationships

- **Is-a →** [[explicit-cot-compression-and-internalization]]: the project is a selective local filler-pruning method within the broader reasoning-compression landscape. Evidence: [[reasoning-pruning-codebase]].
- **Contrasts with →** [[explicit-cot-compression-and-internalization]]: the project's current endpoint preserves useful explicit reasoning rather than fully internalizing or replacing it with latent states. Evidence: [[reasoning-pruning-codebase]], [[from-explicit-cot-to-implicit-cot]], [[training-large-language-models-to-reason-in-a-continuous-latent-space]].

## Boundaries and Failure Modes

- D must remain a judge, never a writer of `target_y`.
- A target that is another goal or intent statement is not a valid useful next step.
- A row with no following unit cannot teach a transition.
- Using a different model family as G breaks the intended self-distillation alignment.
- More aggressive removal is not automatically better; it can erase necessary scratchpad computation or make contexts incoherent.
- Claims of improved efficiency require trained-model evaluation, not only shorter constructed examples.

## Open Questions or Tensions

- How often do current Gemma-4 traces contain removable filler under a strict human audit?
- Does one learned local skip reduce future filler, or does the model generate new filler in response to the pruned context?
- How does multi-depth pruning affect correctness and trace coherence across GSM8K, MATH-Hard, AQUA-RAT, and BBH logical deduction?
- Which redundancy categories require global dependency information rather than local semantics?
- What portion of latency is actually saved when useful explicit reasoning remains?

## Sources

- [[reasoning-pruning-codebase]] — authoritative project contract and current configuration state.
- [[from-explicit-cot-to-implicit-cot]] — removal curriculum, left-side removal, and efficiency-frontier implications.
- [[graph-based-chain-of-thought-pruning]] — global structural view of redundant reflection.
- [[beyond-a-better-planning-with-transformers]] — process-first, compress-later evidence from planning.
- [[training-large-language-models-to-reason-in-a-continuous-latent-space]] — latent-reasoning contrast and planning boundary.
