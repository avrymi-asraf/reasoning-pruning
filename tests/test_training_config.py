"""Tests for structured Gemma training configuration.

These tests ensure the local YAML config still points at the expected Gemma 4
base model, Hub artifacts, and W&B reporting path. They run locally under
uv/pytest as a lightweight guard before submitting the remote HF Jobs trainer.
"""

from pathlib import Path

from reasoning_pruning.training_config import load_training_config


def test_training_config_names_gemma4_and_hub_artifacts():
    config = load_training_config(Path("configs/train/training_gemma4_gsm8k_100.yaml"))

    assert config.base_model == "google/gemma-4-E2B-it"
    assert config.hub_model_id == "avreymi/gemma-4-E2B-it-reasoning-pruning-gsm8k-100-r1"
    assert config.hub_dataset_id == "avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r1"
    assert config.max_steps == 200
    assert config.report_to == ["wandb"]
