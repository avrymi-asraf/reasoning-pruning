# Incremental Generation: Discard-and-Retry

## Problem

The current data-creation loop asks G to generate a full reasoning trace (up to 512–1024 tokens) before asking D whether any span is removable. Most of those tokens are wasted: D only needs to see the first 1–2 units to find filler, and the rest of the trace is discarded once a row is emitted. Expensive for G; also expensive for D when it has to evaluate a long unit list.

## Solution

Generate at most `max_units_per_batch` units (default 2) per G call. Send those units immediately to D.

- If D finds a valid removal → emit a row. Done for this depth.
- If D finds no valid removal → **discard those units entirely**, regenerate from the same context with a fresh sample, and try again.
- Repeat up to `max_retries_per_depth` times. If all attempts fail, stop for this question.

The context never changes between retries within a depth. Because G uses temperature sampling, each attempt produces different units.

## Why Discard, Not Accumulate

Accumulating units that D didn't prune is tempting but wrong: those units may themselves be filler. Accumulating them silently poisons the accepted context and makes later depths train on a dirty prefix. Discarding and retrying keeps the context clean and forces G to produce a version that D can act on immediately.

## Efficiency

With defaults `max_retries_per_depth=3`, `max_units_per_batch=2`, `max_new_tokens=100`:

| Case | G tokens per depth | D units per call |
|------|--------------------|-----------------|
| Success on attempt 1 | ~100 | 2 |
| Success on attempt 2 | ~200 | 2 |
| All attempts fail | ~300 | 2 |
| **Old approach** | **512–1024** | **6–15** |

Expected G cost assuming 60% first-attempt success rate: ~140 tokens (vs 512–1024 = **3–7× savings**).

## Data Contract

Unchanged. `target_y` is still `generated_units[end + 1]` from the successful G trace. The failed retry traces are never stored. The row `metadata` gains a `retry_attempts` field (1 = succeeded on first try) for analysis.

## Live stopping (Transformers)

For Transformers models, generation halts the moment G writes its N-th `\n` character. This is implemented via `StoppingCriteria` — a callback called after every token. Each newline marks the end of one numbered reasoning step, so stopping at newline N = stopping after N complete units. G never even starts writing unit N+1.

For Gemini G, live stopping is not available via the REST API. The prompt requests exactly `max_units_per_batch` numbered lines and `max_new_tokens` (as `maxOutputTokens`) acts as the budget cap. If any provider returns more than `max_units_per_batch` split units, the attempt is discarded rather than truncating it; this preserves the invariant that `generated_units` is a split of the exact stored `generated_trace`.

## Parameters

| Config field | Default | Purpose |
|---|---|---|
| `max_units_per_batch` | `2` | Units to take from each G generation |
| `max_retries_per_depth` | `3` | Max G+D attempts per depth before giving up |
| `generation.max_new_tokens` | `100` | Token budget per G call |

Tune `max_retries_per_depth` upward if yield is too low; tune `max_new_tokens` upward for harder datasets where reasoning steps are longer.
