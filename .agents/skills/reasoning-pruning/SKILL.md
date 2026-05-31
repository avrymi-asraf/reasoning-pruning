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

- `original_trace`: the full raw text G generated for this depth's context, before any splitting or pruning.
- `input_x`: original question plus the useful reasoning prefix.
- `target_y`: the next useful reasoning step after the skipped unit/span.
- `pruning_depth`: the iteration depth for that original question.
- `metadata`: generator model/revision, source dataset/revision, decision model/config, removed span/indexes, generation/pruning config, code version, and round id.

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

Important modules:

- `dataset_builder_config.py`: loads only dataset-builder YAML such as
  `configs/dataset_builder_smoke.yaml`.
- `training_config.py`: loads only training YAML such as
  `configs/training_gemma4_smoke.yaml`.
- `question_source.py`: loads local test questions or HF Dataset source
  questions before generation starts.
- `reasoning_units.py`: simple configurable splitting.
- `pruning_decision.py`: decision result contract.
- `trace_generation.py` and `model_clients.py`: generator and decision model boundaries.
  `model_clients.py` supports Transformers providers and Gemini REST providers;
  use `gemini-json` with `gemini-3.1-flash-lite` for the current D model.
- `data.py`: row builder and prompt/completion conversion for training.
- `pt_dataset_builder.py`: the iterative automatic data-generation engine.
- `hf_dataset_publisher.py`: pushes canonical/training splits as a Hugging Face Dataset.
- `ui_or_cli.py`: exposes workflow commands.
- `model_registry.py`: builds accepted-checkpoint lineage/model-card records.

For `google/gemma-4-E2B-it` smoke training on Hugging Face Jobs:

- Keep dataset-building config and training config separate. Dataset generation
  uses `dataset_builder_smoke.yaml`; training uses `training_gemma4_smoke.yaml`
  and should consume a published Hub dataset.
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

Core self-distillation principle:

The model being trained (Gemma-4-E2B-it) IS the generator G. It generates its own
reasoning traces, D identifies the first genuinely redundant unit, and those PT rows
train the same model to skip its own bad habits. Using a different model as G breaks
this entirely — the trained model learns nothing about its own reasoning patterns.

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
(`model_clients.py`) loads from this directory via `_load_prompt_template(version,
prompts_dir)`. HF Jobs scripts download the same files from the public Hub dataset
`avreymi/reasoning-pruning-prompts` using `hf_hub_download`. To create a new prompt
version: add `prompts/<new-version>.txt`, push to Hub, then reference it in the
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

Current dataset-builder configs (in `configs/data/`):

- `dataset_builder_gsm8k_100_gemma4.yaml`: Gemma-4-E2B-it as G (self-generated,
  private repo), Gemini Flash Lite as D, 100 GSM8K questions. Target:
  `avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r1`. Last completed: 33 rows,
  2026-05-28. Re-running 2026-05-31 (HF job 6a1c32f73a4b8cae6044f163) to add
  `original_trace` field.
- `dataset_builder_gsm8k_200_gemini.yaml`: direct HF source from `openai/gsm8k`, 200 q.
- `dataset_builder_gsm8k_200_text_gemini.yaml`: local text source, 200 questions.
- `dataset_builder_text_gemini.yaml`: local manual question file.
- `dataset_builder_smoke.yaml`: no-source local smoke config.

Current training configs (in `configs/train/`):

- `training_gemma4_gsm8k_100.yaml`: trains on GSM8K-100 dataset, pushes to
  `avreymi/gemma-4-E2B-it-reasoning-pruning-gsm8k-100-r1`. 200 steps, batch 1×4,
  max_length 1024. Active config for current run.
- `training_gemma4_smoke.yaml`: historical — 3 steps, completed 2026-05-28.

Training script is `scripts/train_pt_dataset_job.py` (model-agnostic). Loads the
training config_name subset via `load_dataset(repo, config_name, split="train")`.
Submitted to HF Jobs with HF_TOKEN and WANDB_API_KEY secrets; pass
`TRAINING_CONFIG=configs/train/<file>.yaml` as env var.

Dataset creation runs on HF Jobs, not locally:

Gemma-4-E2B-it is too large (partially offloads to CPU on an 8GB GPU) for fast local
inference over 100 questions. `scripts/create_dataset_gemma4_job.py` is the self-contained
PEP 723 uv job script for this. It requires `HF_TOKEN` and `GEMINI_API_KEY` as secrets.
Load the model with `torch_dtype=torch.bfloat16` and `device_map="auto"` to fit in GPU VRAM.

The local `.env` format may use `export KEY=value`; the CLI loader supports it
and should never print secret values. `GEMINI_API_KEY` is valid as of 2026-05-28.
