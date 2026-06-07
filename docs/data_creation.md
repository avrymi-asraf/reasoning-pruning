# Data creation

Data creation is intentionally small in code but intentionally explicit in docs. The full algorithm lives in `src/reasoning_pruning/data_creation.py`; provider-specific calls live in `src/reasoning_pruning/clients.py`; the CLI wrapper lives in `src/reasoning_pruning/cli.py`.


## Data Creation Contract — Do Not Weaken This

Data creation is not "generate a nicer answer" and it is not "ask D to rewrite G". It is a very specific local-transition extraction process:

```text
input_x = current pruned context + useful prefix from G's newest trace
target_y = the single G-written unit immediately after the skipped filler span
```

D is only a judge. D may say which span is removable, but D must never create, paraphrase, summarize, or repair the target. The target is copied exactly from `generated_units[removed_end_index + 1]`.

The row is valid only if all of these are true:

1. G generated the full `generated_trace` from `context_before_generation`.
2. `generated_units` is a split of that exact G trace.
3. D selected a removable span inside `generated_units`.
4. A following unit exists after the removed span.
5. The following unit is real reasoning: computation, derived fact, or logical deduction.
6. `input_x` contains the old context plus only the useful G units before the removed span.
7. `target_y` is exactly the following useful G unit.
8. The next depth context is exactly `input_x + "\n" + target_y`.

If any of those fail, do not emit a row for that depth.

### Minimal formula

For one question at depth `d`:

```python
context = format_context(question, accepted_units)
for retry_attempts in range(1, max_retries_per_depth + 1):
    trace = G.generate_reasoning(question=question, context=context)
    generated_units = split_reasoning_units(trace.text)
    if not 2 <= len(generated_units) <= max_units_per_batch:
        continue
    decision = D.find_first_removable_span(question, context, generated_units)
    if decision.valid_for(generated_units):
        break
else:
    stop_this_question()

start = decision.removed_start_index
end = decision.removed_end_index
useful_prefix = generated_units[:start]
target_y = generated_units[end + 1]
input_x = context + ("\n" + "\n".join(useful_prefix) if useful_prefix else "")
accepted_units = accepted_units + useful_prefix + [target_y]
```

The training row teaches only this transition:

```text
input_x -> target_y
```

### Full worked example

Question:

```text
A notebook costs $5. Mina buys 4 notebooks. What is the total cost?
```

Depth 0 starts with no accepted units:

```python
accepted_units = []
context_before_generation = "Question:\nA notebook costs $5. Mina buys 4 notebooks. What is the total cost?"
```

G generates this trace:

```text
1. We need to find the total cost.
2. Each notebook costs $5 and Mina buys 4 notebooks.
3. 5 x 4 = 20.
4. The total cost is $20.
```

After splitting:

```python
generated_units = [
    "We need to find the total cost.",
    "Each notebook costs $5 and Mina buys 4 notebooks.",
    "5 x 4 = 20.",
    "The total cost is $20.",
]
```

D may remove index `0` because it only restates the goal. The next unit, index `1`, is a concrete fact from the problem and can be the target:

```python
removed_start_index = 0
removed_end_index = 0
useful_prefix = []
target_y = generated_units[1]
```

The canonical row is therefore:

```python
input_x = "Question:\nA notebook costs $5. Mina buys 4 notebooks. What is the total cost?"
target_y = "Each notebook costs $5 and Mina buys 4 notebooks."
metadata["removed_span"] = ["We need to find the total cost."]
```

Then update accepted units:

```python
accepted_units = ["Each notebook costs $5 and Mina buys 4 notebooks."]
```

The next depth context must be exactly:

```text
Question:
A notebook costs $5. Mina buys 4 notebooks. What is the total cost?
Each notebook costs $5 and Mina buys 4 notebooks.
```

That is the same as:

```python
row["input_x"] + "\n" + row["target_y"]
```

At the next depth, G continues from this pruned context. If G generates:

```text
1. Now multiply the price by the number of notebooks.
2. 5 x 4 = 20.
3. The total cost is $20.
```

D must **not** remove index `0` if index `1` did not contain actual reasoning. In this example index `1` does contain computation, so D may remove index `0`, producing:

```python
input_x = "Question:\nA notebook costs $5. Mina buys 4 notebooks. What is the total cost?\nEach notebook costs $5 and Mina buys 4 notebooks."
target_y = "5 x 4 = 20."
```

The model is learning to skip filler and jump to the next useful step it already knows how to write.

### Invalid rows that must never be created

Do not create a row when:

- the removed span is the last unit, because there is no `target_y`;
- the would-be `target_y` is another goal statement such as "Now calculate the total";
- D suggests replacement text instead of indices into G's units;
- the row target comes from D, a human, or post-processing rather than from G;
- `input_x + "\n" + target_y` would not equal the next context used for depth `d + 1`;
- G is the base model (`google/gemma-4-E2B-it`) when the fine-tuned model (`avreymi/gemma-4-E2B-it-reasoning-pruning`) is available — after round 1, G must always be the fine-tuned model.

## Data Creation Workflow

The code is smaller now, but the workflow is still explicit. Both the local CLI and the HF Jobs script enter the same shared path:

```text
scripts/create_dataset_gemma4_job.py       (HF Jobs)
  or
scripts/reasoning_pruning_cli.py build-dataset / inspect-dataset
  │
  └── src/reasoning_pruning/cli.py         local CLI wiring only
        │
        ├── data_creation.load_data_creation_config(...)
        │     └── reads configs/data/*.yaml into DataCreationConfig
        │
        ├── data_creation.load_questions(config, hf_token=...)
        │     ├── local_file → .txt / .jsonl questions
        │     └── hf_dataset → datasets.load_dataset(...), field extraction, limit
        │
        ├── clients.create_generator_from_config(config.generator, config.generation)
        │     ├── provider=transformers → TransformersGenerator
        │     └── provider=gemini       → GeminiGenerator
        │
        ├── clients.create_decision_model_from_config(config.decision, config.pruning)
        │     ├── provider=transformers-json → TransformersDecisionModel
        │     └── provider=gemini-json       → GeminiDecisionModel
        │
        └── data_creation.build_pt_dataset(questions, G, D, config)
              │
              │  for each question and depth:
              │
              ├── format_context(question, accepted_units)
              ├── retry from the same context up to max_retries_per_depth:
              │     ├── G.generate_reasoning(question=question, context=context)
              │     ├── split_reasoning_units(trace.text, strategy=config.unit_split_strategy)
              │     ├── require 2..max_units_per_batch units
              │     ├── D.find_first_removable_span(question, context, generated_units)
              │     └── discard the attempt unless decision.valid_for(generated_units)
              ├── build_pruning_transition_row(...) from the successful attempt only
              ├── advance_context_units(accepted_units, generated_units, decision)
              └── repeat until retries fail or limits stop the loop
```

Publishing is part of `data_creation.py` too: `push_pt_dataset_to_hub(...)` turns rows into a `canonical` config and a `training` config in the same HF dataset repo.

## Qualitative inspection — required when the pipeline shape changes

Normal tests prove the code still runs; they do not prove the pruning data makes sense. Any change to the data-creation structure, public loop functions, unit splitting, prompt contract, client wiring, or context-advance logic must be checked with the qualitative inspection path before treating the change as safe. The goal is to inspect whether G, D, the selected removable span, `target_y`, and the next context still match the project contract.

The shared inspection entry point is `run_qualitative_pruning_inspection(...)` in `src/reasoning_pruning/qualitative_inspection.py`. It intentionally mirrors the production loop but prints each stage:

- original question;
- context before generation;
- G's generated reasoning trace;
- split reasoning units with indices;
- D's pruning decision;
- removed sentence/span;
- selected target sentence copied from G;
- final `input_x -> target_y` training row;
- next context used for the following depth.

Run it from the command line for quick checks:

```bash
uv run python scripts/qualitative_pruning_inspection.py \
    --config configs/data/qualitative_inspection_gemma4_api.yaml \
    --question-index 3 --max-depth 2 --max-retries 2
```

`configs/data/qualitative_inspection_gemma4_api.yaml` uses hosted `gemma-4-26b-a4b-it` through the Gemini API as a cheap Gemma-family proxy G. This config is **only** for qualitative inspection; do not publish or train from its rows because G is not the active fine-tuned self-distillation model. Production dataset creation must still use the current fine-tuned G, currently `avreymi/gemma-4-E2B-it-reasoning-pruning`.

The notebook inspection cell in `notebooks/data_creation_playground.ipynb` must call the same shared helper instead of carrying a separate hand-copied loop. That uniformity is important: when the production loop changes, the script and notebook should show the same fields and preserve the same `input_x + "\n" + target_y` next-context invariant.

### The core idea — G is always the current fine-tuned model

G must be the most recently trained fine-tuned model from this repo, never the base model after round 1. Self-distillation is iterative:
- Round 1: G = `google/gemma-4-E2B-it` (base) → trained → `avreymi/gemma-4-E2B-it-reasoning-pruning`
- Round 2+: G = `avreymi/gemma-4-E2B-it-reasoning-pruning` → trained → next checkpoint

G generates its own reasoning trace from the current context. D does **not** write replacement reasoning; D only marks the first span that is genuinely redundant and safe to skip. The dataset row then teaches G: given the question and the useful prefix, jump directly to the next useful reasoning step that G already wrote.

The training example shape is always:

```text
question + useful reasoning prefix -> next useful reasoning step
```

Using the base model as G in round 2+ generates traces shaped by the base model's habits. Training those rows into the fine-tuned model teaches it to skip patterns it never produces — the data would be misaligned. G must always be the same model that will be trained on the resulting rows.

### One question, one depth, step by step

#### 1. Load config and questions

`src/reasoning_pruning/data_creation.py` reads a YAML file such as `configs/data/dataset_builder_spectrum_gemma4.yaml`. The config says:

- which generator G to use — always the current fine-tuned model, e.g. `avreymi/gemma-4-E2B-it-reasoning-pruning`;
- which decision model D to use, e.g. `gemini-3.1-flash-lite`;
- where questions come from, either a local file or an HF Dataset;
- how many depths/examples to create;
- where to publish the resulting Hub dataset.

The single PT source is `avreymi/reasoning-spectrum-qa` (split `data`, 1000 diverse QA across 6 reasoning families: factual, commonsense, science, arithmetic, multihop, extractive_control). For a local `.txt` question file, each non-empty line is one question. For a local `.jsonl` file, `source_question_field` selects the question text. For the spectrum dataset, `format_spectrum_question` assembles the full prompt body — `context` + `question` + `choices` — because many rows are unanswerable from `question` alone. The answer fields (`gold_answer`, `gold_answer_label`, `reference_solution`, `supporting_facts`) are never shown to G.

#### 2. Build the current context

At depth 0, there are no accepted useful units yet:

```python
accepted_units = []
context = "Question:\n<question text>"
```

At later depths, the context is the original question plus all accepted useful units so far:

```text
Question:
<question text>
<accepted unit 0>
<accepted unit 1>
...
```

This context is exactly what G sees when it is asked to continue reasoning.

#### 3. Ask G to generate reasoning

`src/reasoning_pruning/clients.py` creates the configured generator client. The generator receives the current context and an instruction to write a short batch of numbered reasoning units, bounded by `max_units_per_batch` and `generation.max_new_tokens`, without telling G which reasoning habits D should prune. Transformers generation stops live after the configured number of newline-terminated units; Gemini relies on the prompt and token budget.

Each depth retries from the exact same clean context up to `max_retries_per_depth`. Attempts with too few or too many units, or with no valid D decision, are discarded completely and never enter a row or the accepted context.

Example raw trace from a successful G attempt:

```text
1. We need to find the total cost.
2. Each item costs $5, and there are 4 items.
3. 4 x 5 = 20.
4. The total cost is $20.
```

G's raw text is stored in `generated_trace`. The dataset target is always copied from this G output; D never invents target text.

#### 4. Split the trace into reasoning units

`split_reasoning_units(trace.text)` turns the raw trace into indexed units. The default `numbered_or_lines` strategy removes simple numbering/bullets and uses lines when there are multiple lines, otherwise it falls back to sentences.

For the trace above:

```python
generated_units = [
    "We need to find the total cost.",
    "Each item costs $5, and there are 4 items.",
    "4 x 5 = 20.",
    "The total cost is $20.",
]
```

The list indices are important because D returns span indices into this exact list.

#### 5. Ask D for the first safe removable span

`src/reasoning_pruning/clients.py` formats `prompts/<prompt_version>.txt` with:

- the original question;
- the current context;
- the numbered `generated_units` list.

D returns JSON like:

```json
{
  "has_removal": true,
  "removed_start_index": 0,
  "removed_end_index": 0,
  "reason": "Step 0 only restates the goal without computing anything.",
  "can_continue_after_skip": true
}
```

`PruningDecision.valid_for(generated_units)` accepts a decision only when:

- `has_removal` is true;
- `can_continue_after_skip` is true;
- start/end indices are in range;
- `removed_end_index + 1` exists, because that next unit becomes `target_y`.

The conservative prompt rule is critical: **the unit immediately after the removed span must contain actual computation, a derived fact, or a logical deduction.** It must not be another goal statement, intent statement, or meta-commentary. If the next unit is not useful, D must return `has_removal=false`.

#### 6. Build the canonical row

`build_pruning_transition_row(...)` creates one canonical row. Suppose D removes index `0`; then the target is index `1`:

```python
{
    "id": "pt-<stable hash>",
    "question": "<original question>",
    "context_before_generation": "Question:\n<question text>",
    "generated_trace": "<full raw G trace>",
    "generated_units": ["..."],
    "input_x": "Question:\n<question text>",
    "target_y": "Each item costs $5, and there are 4 items.",
    "pruning_depth": 0,
    "metadata": {
        "generator_model": "google/gemma-4-E2B-it",
        "generator_model_revision": None,
        "decision_model": "gemini-3.1-flash-lite",
        "removed_span": ["We need to find the total cost."],
        "removed_start_index": 0,
        "removed_end_index": 0,
        "decision_reason": "Step 0 only restates the goal...",
    },
}
```

The two most important fields are:

- `input_x`: what the model sees during training/inference — the question plus the useful prefix accepted so far;
- `target_y`: what the model should generate next — the first useful G-written step after the skipped span.

If D removed a later span, `input_x` would include all units before that span because those units are treated as useful prefix.

#### 7. Advance context and repeat

After a row is created, `advance_context_units(...)` updates the accepted useful units:

```python
accepted_units = previous_accepted_units + units_before_removed_span + [target_y]
```

That means the next context is:

```text
Question:
<question text>
<previous accepted units>
<units before removed span>
<target_y>
```

The invariant must always hold:

```python
next_context == row["input_x"] + "\n" + row["target_y"]
```

This invariant is what keeps multi-depth pruning coherent. The loop then asks G to continue from the pruned context and repeats until D finds no valid skip or the configured limits stop the run.

#### 8. Publish canonical and training configs

`push_pt_dataset_to_hub(...)` publishes two configs to the same HF dataset repo:

- `canonical`: the source rows with `input_x`, `target_y`, `generated_trace`, `generated_units`, and metadata;
- `training`: the same rows plus TRL-friendly fields:

```text
prompt     = "{input_x}\n\nContinue with the next useful reasoning step:"
completion = target_y
```

These are Hub dataset **configs** (`config_name="canonical"` and `config_name="training"`), not separate splits, because the schemas differ.

### Common ways agents break this

- Letting D write or rewrite `target_y`. Never do this; `target_y` must come from G's `generated_units`.
- Removing a span when there is no following useful unit. That creates no training target.
- Treating a goal statement like "calculate the total" as useful target reasoning. The target must do actual computation, deduction, or state a concrete derived fact.
- Updating context with the removed filler. The next context keeps only useful prefix units plus `target_y`.
- Using a different generator model than the model family being trained. That breaks the self-distillation goal.
