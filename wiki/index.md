# Wiki Index

## Scope

This wiki accumulates evidence and synthesis about making language-model reasoning more efficient, with special attention to the reasoning-pruning project's conservative local-transition self-distillation method. It distinguishes concise explicit reasoning, full implicit chain-of-thought, structural trace pruning, and latent reasoning rather than treating all token reduction as the same objective. It records research implications without weakening the repository's data-creation contract.

## Topic Map

### Project and Research Direction

- [[reasoning-pruning-project]] — What exactly does the project train, how does it differ from neighboring methods, and which experiments should come next?

### Reasoning Efficiency Methods

- [[explicit-cot-compression-and-internalization]] — How do selective filler pruning, structural trace pruning, Stepwise Internalization, and continuous latent reasoning differ?

## Open Tensions and Contested Claims

- It is unknown how much of current Gemma-4 reasoning is truly removable filler versus useful scratchpad computation or semantically plausible but redundant reflection.
- Stepwise Internalization supports gradual removal in its studied settings, but it is not established that increasing multi-depth local pruning will have the same optimization dynamics in this project.
- Concise explicit reasoning preserves auditability, while full implicit or latent reasoning may offer larger speed gains; the right accuracy–latency–interpretability trade-off remains task-dependent.
- Dependency-aware pruning may find more redundancy than a local judge, but it also introduces a harder and potentially less reliable labeling problem.

## Knowledge Gaps and Next Investigations

- Build a human-audited sample of D decisions across GSM8K, MATH-Hard, AQUA-RAT, and BBH logical deduction.
- Measure trained-model answer accuracy, emitted reasoning tokens, latency, and residual filler across pruning depths.
- Compare local-transition training against direct final-answer training and direct concise-trace training.
- Determine whether repeated verification and low-impact reflection require global dependency analysis.
- Investigate whether planning-heavy tasks expose a boundary where concise linear text is less effective than a richer latent or search-oriented interface.

## Recent Material Updates

- 2026-06-03 — Ingested [[from-explicit-cot-to-implicit-cot]] and connected it to [[reasoning-pruning-project]]; clarified that Stepwise Internalization removes useful CoT toward implicit reasoning, while this project preserves useful explicit G-written steps.
- 2026-06-03 — Added [[explicit-cot-compression-and-internalization]] with supporting sources on latent reasoning, graph-based pruning, and process-first planning compression; identified removal curriculum, structural redundancy, and accuracy–efficiency evaluation as high-value research directions.
