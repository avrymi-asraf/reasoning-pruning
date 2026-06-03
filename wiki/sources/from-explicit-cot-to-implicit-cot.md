---
title: "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step"
type: source
source_kind: paper
author_or_origin: "Yuntian Deng, Yejin Choi, and Stuart Shieber"
published: 2024-05-23
captured: 2026-06-03
url_or_location: "From Explicit CoT to Implicit CoT - simple.html; https://arxiv.org/abs/2405.14838"
reliability: primary
status: processed
tags: [chain-of-thought, implicit-reasoning, reasoning-efficiency, curriculum-learning]
---

# From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step

## Scope and Relevance

This paper studies whether a model trained with explicit chain-of-thought (CoT) supervision can retain reasoning accuracy after the intermediate CoT tokens are progressively removed from its training inputs and outputs. It is directly relevant to reasoning pruning because it treats explicit reasoning tokens as a removable computational interface and shows that the *schedule* of removal matters. It does not study semantic identification of filler, on-policy self-distillation, or preservation of a human-readable concise reasoning trace.

## Faithful Summary

The authors introduce **Stepwise Internalization (ICoT-SI)**. Training begins from an explicit-CoT model, then removes an increasing number of CoT tokens from the beginning of each training example while continuing to finetune the model on the remaining sequence. The intended outcome is implicit CoT: the model produces the final answer without emitting intermediate reasoning tokens, while hidden states carry enough computation to preserve much of the benefit of CoT supervision.

The paper presents removal as a curriculum rather than a one-shot direct-answer objective. It reports that abrupt objective changes can destabilize training, and introduces two stabilizers: resetting optimizer state whenever another token is removed, and **Removal Smoothing**, which occasionally trains on slightly more removal than the current schedule requires. The authors also find that removing tokens from the beginning is substantially more effective than removing from the end.

The reported experiments cover synthetic multi-digit multiplication and GSM8K. The paper reports near-perfect 9-by-9 multiplication accuracy for GPT-2 Small after full internalization, and over 50% GSM8K accuracy for Mistral 7B without explicit intermediate steps. It also reports an accuracy–speed frontier: partial internalization can be useful even when full internalization is too difficult.

## Extracted Knowledge

- **Definition/Object:** Implicit chain-of-thought reasoning — reasoning that may use explicit CoT supervision during training but does not emit natural-language intermediate steps during generation.
- **Method:** Stepwise Internalization — progressively remove CoT tokens and finetune, beginning with a model already trained for explicit CoT.
  - Support: The method is evaluated on multiplication and GSM8K with GPT-2, Phi-3, and Mistral-family models.
  - Scope/conditions: The paper's experiments use task-specific or augmented CoT data; broader task and trace diversity are left for future work.
  - Status: observed.
- **Claim:** Gradual removal can preserve substantially more reasoning ability than training a direct-answer model without CoT supervision.
  - Support: Reported multiplication and GSM8K comparisons against No-CoT baselines.
  - Scope/conditions: Demonstrated on the paper's selected tasks, models, datasets, and training regimes.
  - Status: observed.
- **Claim:** The amount of internalized CoT exposes a useful accuracy–speed trade-off rather than only a binary explicit-versus-implicit choice.
  - Support: Intermediate checkpoints on 11-by-11 multiplication retain useful accuracy while running faster than explicit CoT.
  - Scope/conditions: The particular frontier is task- and model-dependent.
  - Status: observed.
- **Claim:** Removal curricula can be unstable when the objective changes too quickly.
  - Support: Ablations report failures without Removal Smoothing, without optimizer reset, and with an aggressive token-removal rate.
  - Scope/conditions: The evidence comes from the paper's training setup, especially 7-by-7 multiplication.
  - Status: observed.
- **Claim:** Left-side removal performs better than right-side removal.
  - Support: The paper's ablation compares removal sides and proposes that early tokens can be internalized across more remaining positions.
  - Scope/conditions: The causal explanation is a hypothesis; the empirical result is specific to the studied setup.
  - Status: observed.
- **Boundary:** Full implicit CoT loses the interpretable intermediate trace and still generally trails explicit CoT in accuracy.
  - Support: The limitations section and GSM8K results explicitly acknowledge both issues.
  - Status: argued and observed.

## Limitations and Failure Modes

- The paper removes tokens by position, not by a semantic judgment that the removed content is redundant or safe to omit.
- Full internalization deliberately eliminates visible reasoning, so it does not preserve auditability or a concise explicit trace.
- Longer CoT chains make the gradual finetuning curriculum expensive, and aggressive removal can cause non-convergence.
- The reported results do not establish that the same method works across broad reasoning domains or modern long-reasoning model traces.
- The claim that hidden states contain the internalized reasoning is behaviorally motivated; the paper identifies probing those states as future work.

## Integration Candidates

- Update [[explicit-cot-compression-and-internalization]] to distinguish semantic filler pruning from positional full internalization.
- Update [[reasoning-pruning-project]] with removal-curriculum, left-side-removal, and accuracy–efficiency evaluation implications.

## Tensions or Contradictions

- The paper treats complete disappearance of explicit reasoning as the desired endpoint, whereas the reasoning-pruning project intentionally preserves useful G-written reasoning steps and removes only judged filler.
- The paper's strongest efficiency gains come with reduced interpretability, while the project currently retains an explicit, inspectable reasoning trajectory.
