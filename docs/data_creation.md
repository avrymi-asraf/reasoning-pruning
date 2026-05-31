# Data Creation: Purpose, Process, and Format

## Why We Create This Data

Most training datasets teach a model what the right answer looks like. This project teaches a model something different: **where to skip inside its own reasoning**.

The observation is that language models often generate redundant reasoning steps — restatements of the goal, commentary on what they are about to do, numbering artifacts, and similar filler. These steps waste tokens and teach the model nothing useful. The question is: can we train the model to stop generating those steps in the first place?

The key constraint is **self-distillation**: the model must learn from its own reasoning traces, not from a cleaner external model. This means:

- We use the **same model that will be trained** (G) to generate the reasoning.
- A separate **decision model** (D) identifies the first skippable step.
- The resulting examples train G to skip its own bad habits — not to imitate someone else.

If we trained on reasoning from a stronger model, we would be doing knowledge distillation, which is a different idea. This project is specifically about a model compressing its own reasoning style.

---

## What We Do Not Do

This is not:
- Summarization training ("here is a long answer, here is a shorter answer").
- Post-hoc editing ("take the full trace and remove the bad parts").
- Fine-tuning on correct answers from an external source.

The model does not learn to edit what it already wrote. It learns to **continue differently** from a given prefix — to skip a step at generation time, not after the fact.

---

## How the Data Is Created

### Step 1 — Generate a reasoning trace

We take a source question (e.g. from GSM8K) and give it to G. G generates a full reasoning trace. Example:

```
Question: If a train travels 60 km in 1 hour, how far does it go in 90 minutes?

Step 0: I need to figure out the speed of the train.
Step 1: The train travels 60 km per hour, so its speed is 60 km/h.
Step 2: 90 minutes is 1.5 hours.
Step 3: Distance = speed × time = 60 × 1.5 = 90 km.
```

### Step 2 — Split into reasoning units

The trace is split into units (by line, numbered step, or sentence). Each unit is one atomic piece of reasoning.

### Step 3 — Decision model finds the first skippable unit

D (Gemini Flash Lite with the `conservative-skip-v1` prompt) inspects the units and looks for the **first** unit that:
- Contains **no** computation, no numeric value, no logical deduction, no new fact.
- Is followed by a unit that **does** contain actual reasoning (math, deduction, or a concrete fact).

In the example above, Step 0 ("I need to figure out the speed") is a goal statement — pure intent, no math. Step 1 is actual reasoning (it derives the speed value). So D marks Step 0 as removable.

The decision model does **not** remove a step if the step after it is also filler. It only removes when the next step is real computation.

### Step 4 — Build the training example

The training example captures the transition: given the question and everything up to the skip point, what is the next useful step?

```
input_x:  "Question:\n...question...\n\nUseful reasoning prefix:\n(empty)"
target_y: "The train travels 60 km per hour, so its speed is 60 km/h."
```

The prefix is `(empty)` here because the skip happens at the very first step — nothing useful came before it. If the skip happened after Step 1, the prefix would be Step 1.

### Step 5 — Repeat from the pruned context

After creating the first example, the process continues. The pruned context (question + useful prefix + target_y) is fed back to G. G generates a new continuation. D looks for the next skip. A new training example is created.

This means one question can produce multiple examples at increasing `pruning_depth` values (0, 1, 2, ...). Each example teaches a different skip point inside the same reasoning chain.

The loop stops when D finds no safe removal, or when `max_pruning_depth` is reached.

---

## Decision Rules (what D enforces)

These rules come from `prompts/conservative-skip-v1.txt` and are strict:

**A step is removable only if ALL of these hold:**
1. It contains no computation, no numeric value, no logical deduction, and no new fact. It is pure filler: a numbering artifact, commentary, restatement of the problem, or statement of intent.
2. The step immediately after it contains ACTUAL reasoning — a numeric computation, a derived fact, or a logical deduction.
3. Removing it leaves the reasoning coherent.

**These are NOT removable:**
- `"Convert 50 minutes to hours."` — goal statement, but the *next* step does the math.
- `"$12 × 50/60 = $10."` — actual computation.
- `"50/60 = 5/6 hours."` — actual math.

**These ARE removable:**
- `"1."` (bare numbering artifact)
- `"Let me think."` (filler)
- `"This follows the standard approach."` (commentary with no content)

---

## How the Data Is Stored

Each training example is stored as a **canonical row** with this structure:

```json
{
  "id": "pt-3f7a1c2d8e9b0a4f",
  "question": "If a train travels 60 km in 1 hour, how far in 90 minutes?",
  "original_trace": "I need to figure out the speed of the train.\nThe train travels 60 km per hour, so its speed is 60 km/h.\n90 minutes is 1.5 hours.\nDistance = speed × time = 60 × 1.5 = 90 km.",
  "input_x": "Question:\n...\n\nUseful reasoning prefix:\n(empty)",
  "target_y": "The train travels 60 km per hour, so its speed is 60 km/h.",
  "pruning_depth": 0,
  "metadata": {
    "source_model": "google/gemma-4-E2B-it",
    "round_id": "gsm8k-gemma4-100-r1",
    "source_dataset": "openai/gsm8k",
    "decision_model": "gemini-3.1-flash-lite",
    "removed_span": "I need to figure out the speed of the train.",
    "removed_start_index": 0,
    "removed_end_index": 0,
    "decision_reason": "Pure intent statement, no computation."
  }
}
```

The fields:

| Field | What it contains |
|---|---|
| `id` | Stable SHA256 hash of round + question + depth + span indices |
| `question` | The original source question |
| `original_trace` | The full raw text that G generated for this depth's context, before splitting or pruning |
| `input_x` | The model input: question + useful prefix up to the skip |
| `target_y` | The target: the next useful step after the removed span |
| `pruning_depth` | 0 = first skip from this question, 1 = second, etc. |
| `metadata.source_model` | G — the model that generated the trace (same model being trained) |
| `metadata.removed_span` | The text that was removed |
| `metadata.decision_model` | D — the model that made the pruning decision |

### The `target_y` constraint

`target_y` must contain **actual computation or deduction**. A row where `target_y` is another goal statement ("Now I will calculate the distance") is invalid. D enforces this before the row is created.

---

## Two Dataset Configs on the Hub

The dataset is pushed to `avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r1` with two configs:

**`config_name="canonical"`** — the raw rows as described above. Used for inspection, analysis, and future re-use.

**`config_name="training"`** — the same rows reformatted for TRL's SFTTrainer:

```json
{
  "prompt": "Question:\n...\n\nUseful reasoning prefix:\n(empty)\n\nContinue with the next useful reasoning step:",
  "completion": "The train travels 60 km per hour, so its speed is 60 km/h."
}
```

The training script (`scripts/train_pt_dataset_job.py`) reads the `training` config directly. The canonical config is not involved in training — it exists for traceability.

---

## Active Configuration

The current dataset job uses:
- **G**: `google/gemma-4-E2B-it` (the model being trained)
- **D**: `gemini-3.1-flash-lite` with `conservative-skip-v1` prompt
- **Source**: `openai/gsm8k`, train split, 100 questions
- **Config**: `configs/data/dataset_builder_gsm8k_100_gemma4.yaml`
- **Hub dataset**: `avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r1`
