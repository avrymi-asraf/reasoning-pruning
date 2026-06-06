---
title: "Training Large Language Models to Reason in a Continuous Latent Space"
type: source
source_kind: paper
author_or_origin: "Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian"
published: 2024-12-09
captured: 2026-06-03
url_or_location: "https://arxiv.org/abs/2412.06769"
reliability: primary
status: processed
tags: [latent-reasoning, chain-of-thought, reasoning-efficiency, planning]
---

# Training Large Language Models to Reason in a Continuous Latent Space

## Scope and Relevance

This paper introduces Coconut, a method that performs intermediate reasoning in a continuous latent space rather than decoding every reasoning state into natural-language tokens. It is relevant as a more radical endpoint on the reasoning-efficiency spectrum than either concise explicit CoT or direct-answer implicit CoT. It does not identify redundant spans in ordinary generated traces.

## Faithful Summary

Coconut uses the language model's last hidden state as a **continuous thought** and feeds that state back as the next input embedding instead of decoding it into a word token. The paper argues that language tokens are partly devoted to textual coherence and may unnecessarily constrain reasoning. It reports that continuous thoughts can represent multiple alternative next steps and can support a breadth-first-search-like pattern on planning-heavy logical reasoning tasks.

## Extracted Knowledge

- **Definition/Object:** Continuous thought — a last-layer hidden state reused directly as the next input embedding instead of being decoded into language.
- **Claim:** Natural-language CoT may be an inefficient or restrictive reasoning medium because many tokens serve textual coherence rather than reasoning.
  - Support: Motivation and experimental comparisons in the paper.
  - Scope/conditions: This does not imply that all natural-language reasoning tokens are redundant.
  - Status: argued.
- **Claim:** Coconut can improve the accuracy–efficiency trade-off on logical tasks requiring substantial search.
  - Support: The paper reports comparisons with CoT and analyzes multi-path behavior.
  - Scope/conditions: Demonstrated on the paper's selected tasks and training setup.
  - Status: observed.

## Limitations and Failure Modes

- Latent thoughts are not directly human-readable, reducing interpretability and auditability.
- The architecture and training path differ from ordinary decoder-only text generation, so the method is not a drop-in replacement for explicit-trace pruning.
- A latent state that supports multiple possible next steps is conceptually different from choosing a single exact next G-written unit as a supervised target.

## Integration Candidates

- Add a latent-reasoning endpoint to [[explicit-cot-compression-and-internalization]].
- Clarify in [[reasoning-pruning-project]] that filler pruning can be useful even if the long-term research direction later explores non-linguistic computation.

## Tensions or Contradictions

- Coconut seeks to avoid early commitment to a single textual reasoning path, while reasoning pruning deliberately trains a specific next textual transition copied from G.
