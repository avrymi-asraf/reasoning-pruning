# /// script
# dependencies = [
#   "accelerate>=0.34.0",
#   "datasets>=4.0.0",
#   "huggingface-hub>=0.30.0",
#   "pyyaml>=6.0.2",
#   "torchvision>=0.20.0",
#   "transformers>=4.47.0",
# ]
# ///

"""Hugging Face Jobs entry point for Gemma-4 PT dataset creation.

This script is intentionally thin: the real algorithm lives in
`reasoning_pruning.data_creation`, while model-provider details live in
`reasoning_pruning.clients`. It loads the dataset-builder YAML, lets environment
variables override job-sized knobs, runs Gemma-4 as generator G and Gemini as D,
and pushes canonical/training configs to the Hub. It runs as a uv PEP 723 script
on Hugging Face Jobs with `HF_TOKEN` and `GEMINI_API_KEY` supplied as secrets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_pruning.clients import create_decision_model_from_config, create_generator_from_config
from reasoning_pruning.data_creation import (
    build_pt_dataset,
    load_data_creation_config,
    load_questions,
    push_pt_dataset_to_hub,
)

DEFAULT_CONFIG = "configs/data/dataset_builder_gsm8k_100_gemma4.yaml"


def main() -> int:
    config_path = Path(os.environ.get("DATA_CREATION_CONFIG", DEFAULT_CONFIG))
    config = load_data_creation_config(config_path)
    config = _apply_env_overrides(config)

    print(f"Loading questions from {config.source_dataset} ({config.source_limit or 'all'} max)...")
    questions = load_questions(config, hf_token=os.environ.get("HF_TOKEN"))

    print(f"Loading generator: {config.generator['model_id']}")
    generator = create_generator_from_config(config.generator, config.generation)
    decision_model = create_decision_model_from_config(config.decision, config.pruning)

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


def _apply_env_overrides(config):
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
    if os.environ.get("SOURCE_LIMIT"):
        source_limit = int(os.environ["SOURCE_LIMIT"])
    else:
        source_limit = config.source_limit

    return replace(
        config,
        source_limit=source_limit,
        hub_dataset_id=os.environ.get("HUB_DATASET_ID", config.hub_dataset_id),
        private=os.environ.get("PRIVATE", str(config.private)).lower() != "false",
        max_pruning_depth=int(os.environ.get("MAX_PRUNING_DEPTH", config.max_pruning_depth)),
        max_examples_per_question=int(
            os.environ.get("MAX_EXAMPLES_PER_QUESTION", config.max_examples_per_question)
        ),
        generator=generator,
        decision=decision,
        generation=generation,
        pruning=pruning,
    )


if __name__ == "__main__":
    raise SystemExit(main())
