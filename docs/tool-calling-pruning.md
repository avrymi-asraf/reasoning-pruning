# Pruning as Learning: Internalizing Tool-Use Reasoning

## Why this exists

The reasoning-pruning idea works when the model's own traces contain genuinely
removable filler. On Gemma that condition is weak: the traces are already lean,
so the decision model D rarely finds a safe span to skip. There is "not much to
improve," and the resulting datasets are thin.

Complex tool calling fixes this by changing the *source data*, not the method.
When a model must call a complicated tool, it first writes a real, sound
reasoning trace — "which tool, which parameters, what values" — and only then
emits the call. That deliberation is exactly what we want to internalize. We
prune it **even though it is correct**, and teach the model to reach the tool
call with less and less thinking, until (ideally) it calls the tool
automatically.

This is the same self-distillation pipeline. Complex tool calling is **one new
type of source data** alongside gsm8k / MATH / aqua_rat / bbh — not a new
project, not a domain flip, not a parallel system.

## The one conceptual change: what counts as a "useful step"

The existing math contract (see `CLAUDE.md` → "Data Creation Contract — Do Not
Weaken This") says: only remove pure filler, and the target must be a
computation, derived fact, or logical deduction — never a goal or intent
statement.

Tool-calling data needs a **second contract mode that coexists with the math
one**. It does not weaken the math contract; it adds a parallel set of rules for
tool tasks:

- **The tool call is a valid `target_y`.** A function invocation is the useful
  action a tool task is built around, so it is a legitimate target — the
  analogue of "the computation" in math.
- **Deliberation reasoning is removable even when it is sound.** "I should use
  `get_weather`", "it takes `location` and `units`", "location is Paris" — these
  are correct, but the whole point is to make the model stop writing them. In
  math we may only delete filler; in tool tasks we may delete genuine
  reasoning, because automaticity *is* the goal.

`CLAUDE.md` must be updated to describe both modes explicitly. The math mode and
the tool mode are selected by which decision prompt and generator prompt a
config uses, not by branching logic buried in the loop.

## The mechanism: remove one layer of thinking per round, then train

The user-chosen dynamic is **gradual across rounds**, and it maps directly onto
the project's existing iterative self-distillation loop:

```text
Round N:
  G = current fine-tuned model
  for each verified tool task:
      G generates: [R1, R2, ..., Rk, TOOL_CALL]
      D removes the first removable thinking element
      target_y = the next useful step (more reasoning now; the TOOL_CALL later)
  train on these rows  ->  next checkpoint

Round N+1:
  G = the checkpoint from round N (now thinks a little less)
  it regenerates shorter traces; D removes the next layer
  train again

... repeat until the tool call is reached with maximum efficiency
    (possibly atomically: query -> TOOL_CALL with no reasoning between).
```

Each round peels exactly one layer because we train between rounds, so the
"remove all the reasoning before the tool, step by step" behavior emerges over
the round sequence — not by sweeping a single trace to zero. As G's traces
shrink round over round, the "next useful step" naturally becomes the tool call
itself, and the deepest end state is `query (+tools) -> TOOL_CALL`.

Because per-round pruning stays shallow (typically `max_pruning_depth: 1`), the
accumulate dynamic in `advance_context_units` is harmless here — we are not
trying to reach zero reasoning inside one generation.

### Optional accelerator (not the default)

If single-element-per-round convergence is too slow, the same row contract
supports a **within-run family sweep**: from one verified trace, emit several
rows that all target the tool call while peeling reasoning off the front
(`input = query + R1..R(k-1) -> TOOL_CALL`, then `query + R1..R(k-2) ->
TOOL_CALL`, … down to `query -> TOOL_CALL`). This is stepwise chain-of-thought
internalization (Deng et al., 2024). It uses `target = generated_units[end+1]`
unchanged — only the choice of `removed_start` differs. Keep it as a togglable
option, not the primary mechanism.

## The one genuinely new component: a gold-call verifier

Pure self-distillation has no ground truth. Without a correctness check we would
train the model to emit **confident, wrong** tool calls. ToolACE provides the
gold call, so we add a verifier:

1. G generates a trace whose final unit is a tool call.
2. Parse that final unit into `(function_name, args)`.
3. Compare against the dataset's gold call: name must match; args must match
   after normalization (ignore arg order, normalize types, allow omitted
   optional params — exact rules are an implementation detail to pin down).
4. Only if it matches do we build pruning rows from the trace. Otherwise
   discard and retry, reusing the existing discard/retry loop
   (`max_retries_per_depth`, see `docs/incremental-generation.md`).

The verifier is the only structurally new idea. Everything else is prompts,
config, and source parsing.

## Source dataset: Team-ACE/ToolACE

Verified against the HF datasets-server (ungated, Apache-style access):

- **Schema:** two fields, `system` and `conversations`.
  - `system` contains the instruction plus the available functions as a JSON
    list — these are the **tools**.
  - `conversations` is a list of `{from, value}` turns. `from` is
    `user` / `assistant` / `tool`.
    - first `user` turn → the **query**
    - first `assistant` turn → the **gold tool call**, formatted as
      `[FunctionName(arg="val", ...)]` (note: some function names contain
      spaces, e.g. `Market Trends API` — the parser must handle that)
- **Why ToolACE:** ungated, purpose-built for *complex* APIs (multi-parameter,
  nested params, parallel calls), and it carries gold calls the verifier needs.
- **Alternative:** Salesforce `xlam-function-calling-60k` has clean
  `query` / `tools` / `answers` fields but is gated (license acceptance + auth).
  Keep as a fallback if ToolACE quality disappoints.

### Complexity filtering is core, not optional

Trivial tool tasks (single zero-arg call) contain no reasoning to prune — the
exact failure that made the Gemma math runs thin. The config must filter the
source for genuinely complex tasks: multi-parameter calls, nested/structured
params, or parallel/multi-call answers. Without this filter the dataset
regresses to the original problem.

## How it lands in the existing code

The pipeline call graph is unchanged
(`configs/data/*.yaml` → `cli.py` → `data_creation.build_pt_dataset` → per-task
loop). The tool mode plugs into four named seams plus one structural change.

### Structural change: carry the task, not just a string

`load_questions` currently returns `list[str]`. The verifier needs the gold call
and the tools next to the query. Introduce a small structured source task that
flows through the pipeline:

```text
SourceTask:
    question: str          # for tool tasks: query + tool definitions
    tools: <parsed> | None # tool schemas (None for math tasks)
    gold_call: <parsed> | None  # gold tool call (None for math tasks)
```

Math tasks leave `tools` and `gold_call` as `None`; the verifier runs only when
`gold_call` is present. Per the project's no-backward-compatibility rule, change
`load_questions` / `build_pt_dataset` / `build_rows_for_question` signatures
everywhere they are called (CLI, HF Jobs script, notebook, tests) rather than
adding a parallel path.

### Seam 1 — source loading (ToolACE adapter)

Add a `source_type` (or a ToolACE-specific extractor) that turns each ToolACE
row into a `SourceTask`: parse `system` for tools, take the first user turn as
the query, parse the first assistant turn as the gold call, apply the
complexity filter, and compose `question = query + rendered tool definitions`.

### Seam 2 — generator prompt (tool-aware mode)

`_generator_prompt` in `clients.py` is currently hardcoded to math
("write each step as a concrete computation, deduction, or fact"). Add a
tool-calling prompt mode that injects the tools and instructs G to reason
step by step and then emit the tool call **in a fixed format on its own final
line**, so the call survives unit splitting as one atomic unit. Select the mode
from config; do not branch on string heuristics.

### Seam 3 — unit splitting (atomic tool call)

`split_reasoning_units` must keep the tool-call line intact (never split JSON or
`[Func(...)]` across units). Add a split strategy / rule that treats a tool-call
line as a single atomic unit.

### Seam 4 — decision prompt (tool mode)

Add `prompts/tool-internalize-v1.txt`: deliberation about tool choice and
parameters is removable even when correct; the tool call is always a valid
useful target; stop when the unit after the removed span is real reasoning or
the tool call. This is the tool-mode analogue of `conservative-skip-v1.txt`.

### Seam 5 — verifier

A small function that parses the final generated unit as a tool call and
compares it to `gold_call` with the normalization rules above. Wired into the
discard/retry loop so only verified traces produce rows.

### Config + docs

- `configs/data/dataset_builder_toolace_*_gemma4.yaml` — ToolACE source,
  complexity filter, tool-mode generator + decision prompts, `G =
  avreymi/gemma-4-E2B-it-reasoning-pruning`, `D = gemini-3.1-flash-lite`,
  shallow `max_pruning_depth`.
- Update `CLAUDE.md` (both contract modes), `docs/data_creation.md`, and the
  Colab notebook (`notebooks/data_creation_playground.ipynb`) to match any
  changed public signatures — the notebook alignment rule applies.

## What stays exactly the same

- The `input_x -> target_y` row schema and `target = generated_units[end+1]`.
- `build_pruning_transition_row`, `advance_context_units`, the
  `input_x + "\n" + target_y == next_context` invariant.
- Canonical + training config publishing (`push_pt_dataset_to_hub`).
- HF Jobs execution model, W&B logging, iterative self-distillation with
  G = the current fine-tune.

## Open questions to settle during implementation

1. **Arg-normalization rules** for the verifier: how strict on types, optional
   params, and float formatting? Start strict (exact after order-insensitive,
   type-normalized compare), loosen only if the yield is too low.
2. **Tool-call surface format** G should emit (ToolACE's `[Func(args)]` vs a
   JSON `tool_call` block). Pick one and make the generator prompt, the atomic
   splitter, and the verifier parser agree on it.
3. **"with skills etc."** in the original brief — confirm this means tool /
   function calling only, or whether skill-style multi-tool tasks are in scope
   later. Treat as out of scope for the first dataset.
4. **Per-round vs within-run sweep** as the default once we see real Gemma
   yields — keep the sweep behind a flag.

## Validation before any HF Jobs run

- `uv run pytest`
- `uv run python -m py_compile src/reasoning_pruning/*.py scripts/*.py`
- Build a tiny ToolACE slice locally through the Colab path (D on Gemini, G on
  Colab GPU) and eyeball: gold-match filtering works, the tool call survives as
  an atomic unit, removed spans are deliberation, and
  `input_x + "\n" + target_y` reconstructs the next context.
