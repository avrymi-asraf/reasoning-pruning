"""Setup for pipeline inspection on a Colab GPU VM.

Run via: colab exec -s <session> -f scripts/colab_inspect_setup.py --timeout 600
Requires HF_TOKEN and GEMINI_API_KEY already set in the kernel env before running.
After completion the kernel holds: generator, config, questions, decision_model,
run_pipeline_inspection — all ready for interactive inspection via stdin exec snippets.
"""

import subprocess
import sys
import os
from pathlib import Path
from dataclasses import replace

_REPO = Path("/content/reasoning-pruning")
_PROMPTS_DIR = _REPO / "prompts"
_CONFIG_PATH = _REPO / "configs/data/dataset_builder_spectrum_gemma4.yaml"

missing = [k for k in ("HF_TOKEN", "GEMINI_API_KEY") if not os.environ.get(k)]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}. Set them in the kernel before running.")

if not _REPO.exists():
    print("Cloning repo...")
    subprocess.run(
        ["git", "clone", "https://github.com/avrymi-asraf/reasoning-pruning.git", str(_REPO)],
        check=True,
    )
else:
    print("Repo exists — pulling latest...")
    subprocess.run(["git", "-C", str(_REPO), "pull"], check=True)

print("Installing dependencies...")
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/huggingface/transformers.git",
        "accelerate>=0.34.0", "datasets>=4.8.5", "pyyaml>=6.0.2",
    ],
    check=True,
)

sys.path.insert(0, str(_REPO / "src"))

from reasoning_pruning.data_creation import load_data_creation_config, load_questions  # noqa: E402
from reasoning_pruning.clients import (  # noqa: E402
    create_generator_from_config,
    create_decision_model_from_config,
)
from reasoning_pruning.pipeline_inspection import run_pipeline_inspection  # noqa: E402

# G comes from config.generator, never hardcoded — swap the model by editing
# config.generator["model_id"] (the same `replace`/dict-edit pattern used for D
# prompts), so this works for any model under investigation, not just Gemma-4.
config = load_data_creation_config(_CONFIG_PATH)
config = replace(config, max_pruning_depth=4, max_examples_per_question=3, source_limit=10, max_units_per_batch=20)

print(f"Loading G from config: {config.generator['model_id']} (3-5 min on first GPU run)...")
generator = create_generator_from_config(config.generator, config.generation, max_units_per_batch=2)
print("G ready:", config.generator["model_id"])

questions = load_questions(config, hf_token=os.environ.get("HF_TOKEN"))
decision_model = create_decision_model_from_config(
    config.decision, config.pruning, prompts_dir=str(_PROMPTS_DIR)
)

print("\n=== READY FOR INSPECTION ===")
print(f"G: {config.generator['model_id']}")
print(f"D: {config.decision['model_id']} | prompt: {config.decision['prompt_version']}")
print(f"unit_split_strategy: {config.unit_split_strategy}")
print(f"Questions: {len(questions)}")
print("\nKernel globals: generator, config, questions, decision_model, run_pipeline_inspection")
print("Run: rows = run_pipeline_inspection(question=questions[0], generator=generator, decision_model=decision_model, config=config)")
