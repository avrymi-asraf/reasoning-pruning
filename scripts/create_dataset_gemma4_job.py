# /// script
# dependencies = [
#   "accelerate>=0.34.0",
#   "datasets>=4.0.0",
#   "huggingface-hub>=0.30.0",
#   "pyyaml>=6.0.2",
#   "torchvision>=0.20.0",
#   "transformers>=4.47.0",
#   "reasoning-pruning-codex @ git+https://github.com/avrymi-asraf/reasoning-pruning.git",
# ]
# ///

"""Hugging Face Jobs entry point for Gemma-4 PT dataset creation.

This script is intentionally thin: the real algorithm lives in
`reasoning_pruning.data_creation`, while model-provider details live in
`reasoning_pruning.clients`. It installs those from GitHub (the package is public),
falls back to a GSM8K-r2 default config when no YAML is found, embeds the
conservative-skip-v1 prompt so prompts/ directory is not needed, and lets
environment variables override every config knob. It runs as a uv PEP 723 script
on Hugging Face Jobs with `HF_TOKEN` and `GEMINI_API_KEY` supplied as secrets.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# No sys.path manipulation needed — installed via PEP 723 deps from GitHub.

from reasoning_pruning.clients import create_decision_model_from_config, create_generator_from_config
from reasoning_pruning.data_creation import (
    DataCreationConfig,
    build_pt_dataset,
    load_data_creation_config,
    load_questions,
    push_pt_dataset_to_hub,
)

# Embed conservative-skip-v1 so the jobs server doesn't need the prompts/ directory.
_CONSERVATIVE_SKIP_V1 = """\
Decision prompt version: {prompt_version}

You are a conservative pruning decision model. Your job: find the first reasoning unit that is pure filler and can be removed without any loss of correctness.

STRICT REMOVAL CONDITIONS — all must hold:
1. The unit contains NO computation, NO numeric value, NO logical deduction, and NO new fact. It is pure filler (e.g. a numbering artifact, a commentary, a restatement of the problem, or a statement of intent).
2. The unit at index removed_end_index+1 — which becomes the training target — contains ACTUAL reasoning: a numeric computation, a derived fact, or a logical deduction. It must NOT be another goal/intent statement ('Determine X', 'We need to find Y', 'Calculate Z', 'Convert A to B').
3. Removing the span leaves the reasoning coherent.

REMOVABLE examples: '1.' (bare numbering), 'Let me think.' (filler), 'This follows the standard approach.' (commentary).
NOT REMOVABLE: 'Convert 50 minutes to hours.' (goal statement — not a computation), 'Determine the rate per minute.' (intent, not math), '$12 × 50/60 = $10.' (actual computation — keep it), '50/60 = 5/6 hours.' (actual math — keep it).

If the next unit after the candidate removal is itself a goal/intent statement, set has_removal=false.

Question:
{question}

Current context:
{context}

Reasoning units:
{reasoning_units}

Return only JSON: has_removal (bool), removed_start_index (int), removed_end_index (int), reason (string), can_continue_after_skip (bool).
Set can_continue_after_skip=true only when the unit at removed_end_index+1 contains actual math, logic, or a concrete fact — never a goal or intent statement.
"""

# Default config: GSM8K 100 questions, round 2 (avreymi/gemma-4-E2B-it-reasoning-pruning as G).
# Override any field via the corresponding env var (see _apply_env_overrides).
_DEFAULT_CONFIG = dict(
    round_id="gsm8k-gemma4-100-r2",
    source_type="hf_dataset",
    source_dataset="openai/gsm8k",
    source_dataset_revision="main",
    source_questions_path=None,
    source_subset="main",
    source_split="train",
    source_question_field="question",
    source_limit=100,
    code_version=None,
    hub_dataset_id="avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r2",
    private=True,
    generator={"provider": "transformers", "model_id": "avreymi/gemma-4-E2B-it-reasoning-pruning"},
    decision={
        "provider": "gemini-json",
        "model_id": "gemini-3.1-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
        "prompt_version": "conservative-skip-v1",
    },
    generation={"max_new_tokens": 100, "temperature": 0.7, "do_sample": True},
    pruning={"conservative": True, "require_following_step": True, "max_output_tokens": 256, "temperature": 0.0},
    max_pruning_depth=1,
    max_examples_per_question=1,
    unit_split_strategy="numbered_or_lines",
    max_retries_per_depth=3,
    max_units_per_batch=2,
)


def main() -> int:
    config_path_str = os.environ.get("DATA_CREATION_CONFIG")
    if config_path_str and Path(config_path_str).exists():
        config = load_data_creation_config(Path(config_path_str))
        print(f"Loaded config from {config_path_str}")
    else:
        config = DataCreationConfig(**_DEFAULT_CONFIG)
        print("No config file found — using built-in GSM8K r2 defaults.")

    config = _apply_env_overrides(config)

    print(f"Loading questions from {config.source_dataset} ({config.source_limit or 'all'} max)...")
    questions = load_questions(config, hf_token=os.environ.get("HF_TOKEN"))

    print(f"Loading generator: {config.generator['model_id']}")
    generator = create_generator_from_config(config.generator, config.generation, max_units_per_batch=config.max_units_per_batch)

    prompt_version = config.decision.get("prompt_version", "conservative-skip-v1")
    prompts_dir = _write_embedded_prompt(prompt_version)
    decision_model = create_decision_model_from_config(config.decision, config.pruning, prompts_dir=prompts_dir)

    rows = build_pt_dataset(
        questions=questions,
        generator=generator,
        decision_model=decision_model,
        config=config,
    )
    if not rows:
        print("No PT rows generated — nothing to push.")
        return 0

    push_pt_dataset_to_hub(
        rows=rows,
        hub_dataset_id=config.hub_dataset_id,
        token=os.environ.get("HF_TOKEN"),
        private=config.private,
    )
    print(f"Pushed {len(rows)} rows to {config.hub_dataset_id}")
    return 0


def _write_embedded_prompt(prompt_version: str) -> str:
    """Write the embedded prompt to a temp directory and return its path."""
    tmpdir = tempfile.mkdtemp()
    Path(tmpdir, f"{prompt_version}.txt").write_text(_CONSERVATIVE_SKIP_V1)
    return tmpdir


def _apply_env_overrides(config: DataCreationConfig) -> DataCreationConfig:
    from dataclasses import replace

    generation = dict(config.generation)
    pruning = dict(config.pruning)
    generator = dict(config.generator)
    decision = dict(config.decision)

    if os.environ.get("GENERATOR_MODEL"):
        generator["model_id"] = os.environ["GENERATOR_MODEL"]
    if os.environ.get("DECISION_MODEL"):
        decision["model_id"] = os.environ["DECISION_MODEL"]
    if os.environ.get("MAX_NEW_TOKENS"):
        generation["max_new_tokens"] = int(os.environ["MAX_NEW_TOKENS"])
    if os.environ.get("GENERATION_TEMPERATURE"):
        generation["temperature"] = float(os.environ["GENERATION_TEMPERATURE"])

    source_limit = int(os.environ["SOURCE_LIMIT"]) if os.environ.get("SOURCE_LIMIT") else config.source_limit
    source_subset = os.environ.get("SOURCE_SUBSET", config.source_subset) or None

    return replace(
        config,
        round_id=os.environ.get("ROUND_ID", config.round_id),
        source_type=os.environ.get("SOURCE_TYPE", config.source_type),
        source_dataset=os.environ.get("SOURCE_DATASET", config.source_dataset),
        source_dataset_revision=os.environ.get("SOURCE_DATASET_REVISION", config.source_dataset_revision),
        source_subset=source_subset,
        source_split=os.environ.get("SOURCE_SPLIT", config.source_split),
        source_question_field=os.environ.get("SOURCE_QUESTION_FIELD", config.source_question_field),
        source_limit=source_limit,
        hub_dataset_id=os.environ.get("HUB_DATASET_ID", config.hub_dataset_id),
        private=os.environ.get("PRIVATE", str(config.private)).lower() != "false",
        max_pruning_depth=int(os.environ.get("MAX_PRUNING_DEPTH", config.max_pruning_depth)),
        max_examples_per_question=int(
            os.environ.get("MAX_EXAMPLES_PER_QUESTION", config.max_examples_per_question)
        ),
        max_retries_per_depth=int(os.environ.get("MAX_RETRIES_PER_DEPTH", config.max_retries_per_depth)),
        max_units_per_batch=int(os.environ.get("MAX_UNITS_PER_BATCH", config.max_units_per_batch)),
        generator=generator,
        decision=decision,
        generation=generation,
        pruning=pruning,
    )


if __name__ == "__main__":
    raise SystemExit(main())
