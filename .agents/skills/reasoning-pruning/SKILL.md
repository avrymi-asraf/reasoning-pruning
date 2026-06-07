---
name: reasoning-pruning
description: Project notes for automatic reasoning-pruning data creation and HF training.
---

# Reasoning-Pruning Project Notes

The real dataset pipeline is automatic. Do not treat hand-written smoke rows or
manual `reasoning_steps + removable_index` calls as the core system.

Canonical pruning-transition rows are shaped as:

```text
question + useful reasoning prefix -> next useful reasoning step
```

In code, preserve the semantic fields:

- `context_before_generation`: the full context string passed to G for this depth (question + accepted units so far). At depth 0 this is just the question.
- `generated_trace`: the full raw text G generated for this depth's context, before any splitting or pruning.
- `generated_units`: the list of reasoning units produced by splitting `generated_trace`.
- `input_x`: `context_before_generation` extended with the useful prefix units from the current generation (units before the removed span). Satisfies the invariant `input_x + "\n" + target_y == context_before_generation at depth d+1`.
- `target_y`: the next useful reasoning step after the skipped unit/span (from G's output, never from D).
- `pruning_depth`: the iteration depth for that original question.
- `metadata`: `generator_model`, `generator_model_revision`, `decision_model`, `removed_span` (list), `removed_start_index`, `removed_end_index`, `decision_reason`.


Data creation non-negotiables:

- D is only a judge; D never writes, rewrites, paraphrases, or repairs `target_y`.
- `target_y` is always copied exactly from `generated_units[removed_end_index + 1]`.
- Emit no row unless the following unit exists and is actual computation, a derived fact, or a logical deduction.
- The next depth context must equal `row["input_x"] + "\n" + row["target_y"]`.
- Keep the full worked example in `AGENTS.md` and `docs/data_creation.md`; it is required project guidance, not optional prose.

The automatic loop is:

1. Load source questions from config. Supported sources are local smoke/test
   files (`txt` or `jsonl`) and Hugging Face Dataset splits using the configured
   question field, split, revision, token, and optional limit.
2. Ask the current generator model version for a short reasoning batch from the current pruned context.
3. Split generated text into reasoning units and require 2..`max_units_per_batch` units.
4. Ask the decision model for the first conservative safe skip.
5. Discard invalid attempts and retry from the unchanged context up to `max_retries_per_depth`.
6. Build one canonical PT row from the successful attempt only.
7. Update the pruned context by keeping useful prefix plus the next useful step.
8. Repeat until retries fail, max depth, max examples, or generation failure.

Keep the detailed walkthrough in `AGENTS.md` and `docs/data_creation.md` in sync with this loop. It is intentionally verbose because future agents need the examples and invariants to avoid corrupting data creation.

Important modules after the data-creation simplification:

- `data_creation.py`: the whole automatic data-creation core. It loads dataset-builder YAML, reads local/HF questions, defines `GeneratedTrace`, `PruningDecision`, and `PruningObserver`, splits reasoning units, runs the single per-question loop `build_rows_for_question`, builds canonical rows, advances context, converts rows to prompt/completion, and publishes canonical/training configs to the Hub.
- `clients.py`: model boundaries for G and D. It supports Transformers and Gemini generators plus Transformers/Gemini JSON decision models, prompt loading, Gemini REST transport, and JSON decision parsing.
- `cli.py`: local CLI wiring for `build-dataset` and `inspect-dataset`; it loads `.env`, builds clients from config, calls `data_creation.build_pt_dataset`, and optionally publishes.
- `qualitative_inspection.py`: a printing `PruningObserver` plus `run_qualitative_pruning_inspection`, which calls the **production** `build_rows_for_question` with that observer. It does NOT reimplement the loop — printing is a hook on the real loop, so script/notebook output can never drift from what data creation runs. (A copied inspection loop once drifted to a ≥2-unit gate while production moved to ≥3; the observer design exists to prevent exactly that.)
- `scripts/qualitative_pruning_inspection.py`: the thin CLI entry point (arg parsing, `.env`, config + client construction) that calls `run_qualitative_pruning_inspection`. Same library-vs-entry-point split as `data_creation.py`/`cli.py` — it is not a second copy of the loop. Run it after any pipeline change to read the printed flow and judge whether G/units/D make sense, not just that tests pass.
- `training_config.py`: loads only training YAML from `configs/train/`
  (e.g. `configs/train/training_gemma4_gsm8k_100.yaml`).
- `model_registry.py`: builds accepted-checkpoint lineage/model-card records.

The old many-file data creation split (`dataset_builder_config.py`, `question_source.py`,
`trace_generation.py`, `reasoning_units.py`, `pruning_decision.py`, `data.py`,
`pt_dataset_builder.py`, `hf_dataset_publisher.py`, `model_clients.py`, and
`ui_or_cli.py`) was intentionally collapsed. Do not recreate those layers unless the
project grows enough to make the extra indirection worth it.

For `google/gemma-4-E2B-it` training on Hugging Face Jobs:

- Keep dataset-building config and training config separate. Dataset generation
  uses `configs/data/dataset_builder_spectrum_gemma4.yaml`; training uses
  `configs/train/training_gemma4_gsm8k_100.yaml` and consumes the published Hub
  dataset.
- Use TRL `SFTTrainer` with a prompt/completion dataset and `completion_only_loss=True`.
- Include `torchvision`; Gemma 4 loads a multimodal processor even for text-only examples.
- Use W&B for run observation, as required by `docs/plan.md`.
- Push the generated PT dataset to the Hugging Face Hub before training; the trainer should load the Hub dataset training split, not build inline smoke examples.
- Push accepted adapter/model artifacts to the Hugging Face Hub and document parent model, PT dataset version, config, W&B run, evaluation metrics, and acceptance reason.
- Avoid Trackio for the current LoRA smoke path; it failed after training while exporting empty `rank_pattern` config metadata to Parquet.

HF publisher notes:

- Use `config_name` (not `split`) when pushing canonical and training to the same
  Hub repo — they have different schemas (`training` adds `prompt`/`completion`),
  and `DatasetDict.push_to_hub` requires matching features across all splits.
  Push each as `Dataset.push_to_hub(..., config_name="canonical", split="train")`
  and `Dataset.push_to_hub(..., config_name="training", split="train")`.
- Load training split downstream with `load_dataset(repo_id, name="training")`.
- `datasets` must be in `pyproject.toml` dependencies for HF-source question loading
  and Hub publishing.

Core self-distillation principle — G is always the current fine-tuned model:

Self-distillation is iterative. G must always be the most recently trained fine-tuned
model from this repo, never the base model after round 1:
- Round 1: G = `google/gemma-4-E2B-it` (base) → trained → `avreymi/gemma-4-E2B-it-reasoning-pruning`
- Round 2+: G = `avreymi/gemma-4-E2B-it-reasoning-pruning` → trained → next checkpoint

Using the base model as G in round 2+ generates traces shaped by the base model's
habits. Training on those rows teaches the fine-tuned model to skip patterns it never
produces — the data would be misaligned. G must always be the same model that will
be trained on the resulting rows.

The active generator model is `avreymi/gemma-4-E2B-it-reasoning-pruning`.

Decision prompt quality rule (incremental-skip-v2, updated 2026-06-03):

`can_continue_after_skip=true` requires that the unit at `removed_end_index+1` contains
ACTUAL computation, a derived fact, or a logical deduction — NOT a goal statement
('Determine X', 'We need to find Y', 'Calculate Z'). Goal statements are themselves
filler. The prompt now contains explicit REMOVABLE vs NOT REMOVABLE examples.

Generator prompt rule:

The G prompt requests a fixed short batch of numbered steps, one per line, but must not tell
G which reasoning habits D should prune. For instruction-tuned models (ending in `-it`), use
`apply_chat_template` with a user message.

No backward compatibility:

When anything changes in this project, change it everywhere. Do not maintain compatibility
shims, re-exports for removed names, or parallel old/new code paths.

Prompts folder:

Decision-model prompt templates live in `prompts/` as `.txt` files named by
`prompt_version` (e.g. `prompts/incremental-skip-v2.txt`). The local package
(`clients.py`) loads from this directory via `load_prompt_template(version,
prompts_dir)`. The HF Jobs script (`create_dataset_gemma4_job.py`) now uses the
shared package code instead of embedding a separate prompt constant. To create a
new prompt version: add `prompts/<new-version>.txt` and reference it in the
dataset-builder config's `decision.prompt_version` field.

Config folder split:

- Dataset-builder configs live in `configs/data/` (`dataset_builder_*.yaml`).
- Training configs live in `configs/train/` (`training_*.yaml`).
- The `TRAINING_CONFIG` env var for HF Jobs must point to `configs/train/...`.
- The `--config` CLI arg must point to `configs/data/...`.

Model-agnostic code rule:

Library code must work with any decoder-only transformer model (Gemma, Llama, Qwen,
Mistral, etc.). Use `AutoModelForCausalLM` and `AutoTokenizer` — never hardcode model
class names. Make model IDs, dataset repos, and training hyperparameters configurable
via YAML or env vars. Scripts may have model-specific names (e.g.
`create_dataset_gemma4_job.py`) but their internal logic must extend cleanly to any
other model by changing only config values.

## Source and dataset-builder config

The single PT source is `avreymi/reasoning-spectrum-qa` (split `data`, 1000 diverse
QA across 6 reasoning families: factual, commonsense, science, arithmetic, multihop,
extractive_control). `format_spectrum_question` in `data_creation.py` assembles the
full prompt per row (`context` + `question` + `choices`); answer fields are never
shown to G. Factual and extractive_control rows rarely yield multi-step reasoning,
so the pruning loop emits few or no rows for them — that is expected.

There is one dataset-builder config in `configs/data/`, using
`avreymi/gemma-4-E2B-it-reasoning-pruning` as G (round 2):

- `dataset_builder_spectrum_gemma4.yaml`: reasoning-spectrum-qa (`source_limit: 200`
  by default; raise for a full round). Target: `avreymi/reasoning-pruning-pt-spectrum-gemma4-r2`.

`create_dataset_gemma4_job.py` installs the package via
`git+https://github.com/avrymi-asraf/reasoning-pruning.git` in PEP 723 deps, embeds the
incremental-skip-v2 prompt (prompts/ dir not available on HF Jobs), defaults to GSM8K r2 config,
and supports all source fields as env vars. Pass `SOURCE_DATASET`, `SOURCE_SUBSET`, `SOURCE_SPLIT`,
`SOURCE_QUESTION_FIELD`, `SOURCE_LIMIT`, `HUB_DATASET_ID`, `ROUND_ID`, `MAX_NEW_TOKENS` to run
any of the 4 datasets without needing the YAML on the server.

Current training configs (in `configs/train/`):

- `training_gemma4_gsm8k_100.yaml`: trains on `avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r1`,
  pushes adapter to `avreymi/gemma-4-E2B-it-reasoning-pruning-gsm8k-100-r1`.
  200 steps, batch 1×4, max_length 1024. Active config.

Training script is `scripts/train_pt_dataset_job.py` (model-agnostic). Loads the
training config_name subset via `load_dataset(repo, config_name, split="train")`.
Submitted to HF Jobs with HF_TOKEN and WANDB_API_KEY secrets; pass
`TRAINING_CONFIG=configs/train/<file>.yaml` as env var.

Dataset creation runs on HF Jobs, not locally:

Gemma-4-E2B-it is too large (partially offloads to CPU on an 8GB GPU) for fast local
inference over 100 questions. `scripts/create_dataset_gemma4_job.py` is the self-contained
PEP 723 uv job script for this. It requires `HF_TOKEN` and `GEMINI_API_KEY` as secrets.
Load the model with `torch_dtype=torch.bfloat16` and `device_map="auto"` to fit in GPU VRAM.

The `build-dataset --dry-run` CLI command and `inspect-dataset` command only work with
configs that use a Gemini or local-file generator, not `transformers` provider, because
Gemma-4-E2B-it can't be loaded locally without a GPU.

## Qualitative inspection for pipeline sanity

When changing the data-creation structure, public loop signatures, client constructors,
unit splitting, D prompt contract, or context-advance behavior, do not rely only on
the automated tests — they only prove the pipeline is wired and runs. The real check
is running the qualitative inspection against a live model and reading the printed
flow (does G make sense? is the unit split clean? is D removing real filler?). The
inspector hooks the production loop via `PruningObserver`, so its output is exactly
what data creation produces:

```bash
uv run python scripts/qualitative_pruning_inspection.py \
  --config configs/data/qualitative_inspection_gemma4_api.yaml \
  --question-index 3 --max-depth 2 --max-retries 2
```

The qualitative config uses hosted `gemma-4-26b-a4b-it` as a cheap Gemma-family proxy
G. It is for inspection only; never publish/train rows from it because production
self-distillation must use the current fine-tuned G. Keep the notebook inspection cell
calling the same shared helper so script and notebook output remain uniform.

## Full multi-depth playground (Google Colab) — the D-prompt playground

`notebooks/data_creation_playground.ipynb` — runs the complete multi-depth pipeline
on Colab GPU. G = Gemma-4 (local on Colab), D = Gemini Flash Lite (cloud). This is the
**only** place to iterate D prompts: there is no cached-trace workflow — G runs live on
the Colab GPU so every prompt change is judged against fresh traces.

**Gemma 4 requires transformers from git main** — not in any stable PyPI release.
The setup cell installs `git+https://github.com/huggingface/transformers.git`. Do not
change this to a PyPI pin; it will break with `KeyError: 'gemma4'`.

Setup: enable GPU runtime (T4 or A100), add `HF_TOKEN` and `GEMINI_API_KEY` to Colab
secrets, then run cells top-to-bottom. The notebook:

1. Clones the repo and installs deps (transformers from git, ~2 min)
2. Initializes G from `avreymi/gemma-4-E2B-it-reasoning-pruning` (~5GB download, ~1-2 min)
3. Loads the spectrum dataset-builder config and pulls real questions via
   `load_questions`, then overrides `max_pruning_depth`/`source_limit` with
   `dataclasses.replace(...)` for quick runs
4. Calls `build_rows_for_question` (single question, full depth loop, verbose output)
   or `build_pt_dataset` (multiple questions, summary)

`build_rows_for_question` is the right entry point for the playground — it is the
same function `build_pt_dataset` calls internally, now exposed as a public API.

**Iterating D prompts (three dedicated cells, no extra deps):**
1. *List / view* — `list_prompts()` shows every `prompts/*.txt`; `show_prompt("<stem>")`
   prints one inline.
2. *Write / edit* — set `PROMPT_NAME` + `PROMPT_TEXT` to create a new version (or
   overwrite an existing file with `OVERWRITE = True`); the cell writes `prompts/<name>.txt`.
   Only `{prompt_version} {question} {context} {reasoning_units}` may be single braces.
3. *Choose* — set `PROMPT_VERSION`; the cell overrides `config.decision["prompt_version"]`
   and rebuilds D via `create_decision_model_from_config(config.decision, config.pruning,
   prompts_dir=PROMPTS_DIR)` — the production path. D therefore uses the **model from
   `config.decision`** (`gemini-3.1-flash-lite`), not a hardcoded model, and runs at
   `config.pruning` temperature `0.0` for fair comparison.

Re-run the build cells to compare removal rate and emitted `input_x → target_y` rows
across versions.

**Notebook alignment:** the notebook calls production library functions directly with no
wrappers. When public signatures in `data_creation.py` or `clients.py` change, or when
the active G model advances to a new checkpoint, update the notebook to match. See the
full alignment rule in `AGENTS.md`.

The local `.env` format may use `export KEY=value`; the CLI loader supports it
and should never print secret values. `GEMINI_API_KEY` is valid as of 2026-05-28.
