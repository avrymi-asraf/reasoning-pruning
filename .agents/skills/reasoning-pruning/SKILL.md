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
2. Ask the current generator model version to generate reasoning for the current pruned context.
3. Split generated text into reasoning units with the configured strategy.
4. Ask the decision model for the first conservative safe skip.
5. Build one canonical PT row.
6. Update the pruned context by keeping useful prefix plus the next useful step.
7. Repeat until no safe skip, invalid next step, max depth, max examples, or generation failure.

Keep the detailed walkthrough in `AGENTS.md` and `docs/data_creation.md` in sync with this loop. It is intentionally verbose because future agents need the examples and invariants to avoid corrupting data creation.

Important modules after the data-creation simplification:

- `data_creation.py`: the whole automatic data-creation core. It loads dataset-builder YAML, reads local/HF questions, defines `GeneratedTrace` and `PruningDecision`, splits reasoning units, builds canonical rows, advances context, converts rows to prompt/completion, and publishes canonical/training configs to the Hub.
- `clients.py`: model boundaries for G and D. It supports Transformers and Gemini generators plus Transformers/Gemini JSON decision models, prompt loading, Gemini REST transport, and JSON decision parsing.
- `cli.py`: local CLI wiring for `build-dataset` and `inspect-dataset`; it loads `.env`, builds clients from config, calls `data_creation.build_pt_dataset`, and optionally publishes.
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
  uses `configs/data/dataset_builder_gsm8k_100_gemma4.yaml`; training uses
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

Decision prompt quality rule (conservative-skip-v1, updated 2026-05-28):

`can_continue_after_skip=true` requires that the unit at `removed_end_index+1` contains
ACTUAL computation, a derived fact, or a logical deduction — NOT a goal statement
('Determine X', 'We need to find Y', 'Calculate Z'). Goal statements are themselves
filler. The prompt now contains explicit REMOVABLE vs NOT REMOVABLE examples.

Generator prompt rule:

The G prompt explicitly says "write each step as a concrete computation, deduction, or
fact — do not write goal statements or intent". For instruction-tuned models (ending in
`-it`), use `apply_chat_template` with a user message.

No backward compatibility:

When anything changes in this project, change it everywhere. Do not maintain compatibility
shims, re-exports for removed names, or parallel old/new code paths.

Prompts folder:

Decision-model prompt templates live in `prompts/` as `.txt` files named by
`prompt_version` (e.g. `prompts/conservative-skip-v1.txt`). The local package
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

## Trace generation (no D model)

`scripts/generate_traces_job.py` is a self-contained PEP 723 HF Jobs script that
generates raw G traces across multiple source datasets without running D. Use it
to build trace caches before applying D decisions.

Submit via CLI (local file path is accepted):
```bash
source .env && .venv/bin/hf jobs uv run \
  --flavor a10g-small \
  --timeout 60m \
  --secrets HF_TOKEN \
  --detach \
  scripts/generate_traces_job.py
```

Then monitor with:
```bash
source .env && .venv/bin/hf jobs logs --tail 40 <job-id>
source .env && .venv/bin/hf jobs logs -f <job-id>   # live stream (blocking)
```

The script reads `GENERATOR_MODEL`, `HUB_OUTPUT`, and `N_PER_SOURCE` from env vars;
defaults produce 25 traces × 4 datasets = 100 traces at `avreymi/reasoning-traces-gemma4-100`.
The output dataset schema: `source_dataset`, `question`, `trace`, `units` (list[str]).

**Important dataset caveats discovered:**
- `lighteval/MATH` does not exist on HF Hub — use `lighteval/MATH-Hard` (field `problem`, split `train`)
- `trust_remote_code=True` is deprecated in recent `datasets` and causes DatasetNotFoundError for
  some repos (e.g. `lighteval/MATH`). Do not pass it — standard Parquet datasets work without it.
- 60-minute timeout is insufficient for 100 traces with BBH logical-deduction questions (which
  produce very long outputs). Use 2h for 100 traces with Gemma-4 on a10g-small.
- Push incrementally after each source batch using `config_name=src_name` so timeouts don't lose
  completed work. The final combined push uses `config_name="all"`.

## Completed trace datasets

- `avreymi/reasoning-traces-gemma4-100` (private) — 100 Gemma-4 raw traces (no D applied).
  25 each from: gsm8k (~7 units/trace), lighteval/MATH-Hard (~28 units), deepmind/aqua_rat (~30 units),
  lukaemon/bbh logical_deduction_seven_objects (~62 units). Each source has its own config_name
  plus a combined `all` config. Generated by `scripts/generate_traces_job.py`.

Current dataset-builder configs (in `configs/data/`) — all use `avreymi/gemma-4-E2B-it-reasoning-pruning` as G (round 2):

- `dataset_builder_gsm8k_100_gemma4.yaml`: 100 GSM8K questions. Target: `avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r2`.
- `dataset_builder_math_25_gemma4.yaml`: 25 competition math problems from `lighteval/MATH-Hard`. Target: `avreymi/reasoning-pruning-pt-math-25-gemma4-r2`.
- `dataset_builder_aqua_rat_25_gemma4.yaml`: 25 algebra word problems from `deepmind/aqua_rat`. Target: `avreymi/reasoning-pruning-pt-aqua-rat-25-gemma4-r2`.
- `dataset_builder_bbh_logical_25_gemma4.yaml`: 25 logical-deduction puzzles from `lukaemon/bbh` (subset `logical_deduction_seven_objects`). Target: `avreymi/reasoning-pruning-pt-bbh-logical-25-gemma4-r2`.

Round 2 data creation jobs (submitted 2026-06-02, all running):
- GSM8K 100: job `6a1f2fdfb2914899801366df` → `avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r2`
- MATH-Hard 25: job `6a1f2fea4e2701e4a03f603c` → `avreymi/reasoning-pruning-pt-math-25-gemma4-r2`
- AQUA-RAT 25: job `6a1f2fedb2914899801366e1` → `avreymi/reasoning-pruning-pt-aqua-rat-25-gemma4-r2`
- BBH logical 25: job `6a1f2fef4e2701e4a03f603e` → `avreymi/reasoning-pruning-pt-bbh-logical-25-gemma4-r2`

HF Jobs job script fix (2026-06-02): `create_dataset_gemma4_job.py` now installs the package via
`git+https://github.com/avrymi-asraf/reasoning-pruning.git` in PEP 723 deps, embeds the
conservative-skip-v1 prompt (prompts/ dir not available on HF Jobs), defaults to GSM8K r2 config,
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

D-prompt playground (local, Gemini as D):

`scripts/generate_traces.py --config <cfg.yaml> --output data/traces/out.jsonl` — loads
questions from the configured source, calls G once per question, saves trace+units to JSONL.
`scripts/replay_decisions.py --traces data/traces/out.jsonl --prompt prompts/<version>.txt` —
runs D on the cached traces with any `.txt` prompt file. Swap and re-run to compare D
prompt versions without regenerating G traces. Requires `GEMINI_API_KEY` in `.env`.

The script also supports loading directly from the Hub (skipping local JSONL):
`scripts/replay_decisions.py --hub [config] --prompt prompts/<version>.txt --limit 10`
`config` is any config_name from `avreymi/reasoning-traces-gemma4-100` (gsm8k, math,
aqua_rat, bbh, or all — defaults to all). Requires `HF_TOKEN` + `GEMINI_API_KEY`.

Output shows the decision AND the actual PT row (input_x, target_y) that would be
emitted — only depth-0 rows since the cached dataset has one raw trace per question.

## Full multi-depth playground (Google Colab)

`notebooks/data_creation_playground.ipynb` — runs the complete multi-depth pipeline
on Colab GPU. G = Gemma-4 (local on Colab), D = Gemini Flash Lite (cloud).

**Gemma 4 requires transformers from git main** — not in any stable PyPI release.
The setup cell installs `git+https://github.com/huggingface/transformers.git`. Do not
change this to a PyPI pin; it will break with `KeyError: 'gemma4'`.

Setup: enable GPU runtime (T4 or A100), add `HF_TOKEN` and `GEMINI_API_KEY` to Colab
secrets, then run cells top-to-bottom. The notebook:

1. Clones the repo and installs deps (transformers from git, ~2 min)
2. Initializes G from `avreymi/gemma-4-E2B-it-reasoning-pruning` (~5GB download, ~1-2 min)
3. Loads the GSM8K dataset-builder config, then overrides `max_pruning_depth` with
   `dataclasses.replace(config, max_pruning_depth=4)` for quick runs
4. Calls `build_rows_for_question` (single question, full depth loop, verbose output)
   or `build_pt_dataset` (multiple questions, summary)

`build_rows_for_question` is the right entry point for the playground — it is the
same function `build_pt_dataset` calls internally, now exposed as a public API.
Change the `prompt_version` in the D init cell to test new prompts; no other changes needed.

**Notebook alignment:** the notebook calls production library functions directly with no
wrappers. When public signatures in `data_creation.py` or `clients.py` change, or when
the active G model advances to a new checkpoint, update the notebook to match. See the
full alignment rule in `AGENTS.md`.

The local `.env` format may use `export KEY=value`; the CLI loader supports it
and should never print secret values. `GEMINI_API_KEY` is valid as of 2026-05-28.
